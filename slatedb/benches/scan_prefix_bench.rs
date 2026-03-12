/// Benchmark comparing three prefix scan strategies:
///   1. scan_prefix WITHOUT prefix bloom filter (baseline)
///   2. scan_prefix WITH prefix bloom filter
///   3. scan_prefix_by_recency WITH prefix bloom filter
///
/// Run: cargo bench -p slatedb --bench scan_prefix_bench
///
/// Design:
/// - Meta cache (filter + index blocks) is enabled and warmed before measurement.
///   Data block cache is disabled (SplitCache with meta_cache only, no block_cache).
///   This means filter/index reads are free after warmup, but every data block read
///   still hits the (throttled) object store.
/// - Object store wrapped with ThrottledStore (5ms per GET) to simulate S3-like latency.
///   Throttling is disabled during population & warmup, re-enabled for measurement.
/// - 100 prefixes spread across 100 flush rounds. Each round writes 10 of 100
///   prefixes (round-robin), so the queried prefix appears in 10 out of 100 SSTs.
/// - Sentinel keys ("aa_*", "zz_*") in every flush ensure all SSTs cover the full
///   key range, preventing free manifest-level range pruning.
/// - Each flush round → 1 L0 SST (l0_sst_size_bytes set large).
/// - DurabilityLevel::Remote ensures only data flushed to object storage is read
///   (memtables are skipped), so the recency iterator must read from SSTs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use slatedb::config::{
    DurabilityLevel, FlushOptions, FlushType, PutOptions, ScanOptions, Settings, WriteOptions,
};
use slatedb::db_cache::SplitCache;
use slatedb::{BloomFilterPolicy, Db, FixedPrefixExtractor};
use tokio::runtime::Runtime;

// ── Parameters ──────────────────────────────────────────────────────────────

/// Total distinct prefixes (4 bytes each: "px00".."px99").
const NUM_PREFIXES: usize = 100;
/// Prefixes written per flush round (round-robin from the 100).
/// With 10 prefixes per flush and 100 rounds, each prefix appears in 10 SSTs.
const PREFIXES_PER_FLUSH: usize = 10;
/// Keys per prefix within a single flush round.
const KEYS_PER_PREFIX: usize = 5;
/// Number of flush rounds → L0 SSTs.
const NUM_FLUSHES: usize = 100;
/// Scan iterations for stable averages.
const SCAN_ITERATIONS: usize = 20;
/// Query prefix that appears in 10 SSTs (rounds 9,19,29,...,99).
/// Round r: start_px = (r * 10) % 100. px95 is included when start_px=90,
/// i.e. rounds 9, 19, 29, 39, 49, 59, 69, 79, 89, 99.
/// Most recent SST with px95 is round 99.
const QUERY_PREFIX: &[u8; 4] = b"px95";
/// Simulated GET latency per call (approximates S3 round-trip).
const GET_LATENCY: Duration = Duration::from_millis(1);

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_key(prefix_idx: usize, suffix: usize) -> Vec<u8> {
    format!("px{:02}{:08}", prefix_idx, suffix).into_bytes()
}

fn make_value(round: usize, suffix: usize) -> Vec<u8> {
    format!("v-{}-{}", round, suffix).into_bytes()
}

fn make_throttled_store() -> Arc<ThrottledStore<InMemory>> {
    // Start with zero latency; we enable throttling after warmup.
    Arc::new(ThrottledStore::new(InMemory::new(), ThrottleConfig::default()))
}

fn enable_throttle(store: &ThrottledStore<InMemory>) {
    store.config_mut(|cfg| {
        cfg.wait_get_per_call = GET_LATENCY;
    });
}

fn disable_throttle(store: &ThrottledStore<InMemory>) {
    store.config_mut(|cfg| {
        cfg.wait_get_per_call = Duration::ZERO;
    });
}

/// Build a SplitCache that caches filter + index (meta) but NOT data blocks.
fn meta_only_cache() -> Arc<SplitCache> {
    use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
    let meta = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
        max_capacity: 256 * 1024 * 1024, // 256 MB — plenty for meta blocks
        ..Default::default()
    }));
    Arc::new(
        SplitCache::new()
            .with_block_cache(None) // no data block caching
            .with_meta_cache(Some(meta))
            .build(),
    )
}

/// Populate the DB with a sparse prefix distribution.
/// Each flush round writes PREFIXES_PER_FLUSH consecutive prefixes (round-robin)
/// so a given prefix appears in 10 out of NUM_FLUSHES SSTs.
///
/// Every round also writes sentinel keys at the start ("aa_sentinel") and end
/// ("zz_sentinel") of the key space. This ensures every SST's key range overlaps
/// with every prefix range, preventing free range-based pruning at the manifest
/// level and forcing the bloom filter / recency iterator to actually do work.
async fn populate_db(db: &Db) {
    let write_opts = WriteOptions {
        await_durable: false,
    };
    let put_opts = PutOptions::default();

    for round in 0..NUM_FLUSHES {
        // Sentinel keys that ensure every SST covers the full key space
        let lo_key = format!("aa_sentinel_{:04}", round).into_bytes();
        let hi_key = format!("zz_sentinel_{:04}", round).into_bytes();
        db.put_with_options(&lo_key, b"s", &put_opts, &write_opts)
            .await
            .unwrap();
        db.put_with_options(&hi_key, b"s", &put_opts, &write_opts)
            .await
            .unwrap();

        let start_px = (round * PREFIXES_PER_FLUSH) % NUM_PREFIXES;
        for i in 0..PREFIXES_PER_FLUSH {
            let px = (start_px + i) % NUM_PREFIXES;
            for s in 0..KEYS_PER_PREFIX {
                let key = make_key(px, round * KEYS_PER_PREFIX + s);
                let val = make_value(round, s);
                db.put_with_options(&key, &val, &put_opts, &write_opts)
                    .await
                    .unwrap();
            }
        }
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
    }
}

/// Warmup: do one full prefix scan to populate the meta cache (filter + index blocks).
/// Uses cache_blocks: true so filter/index get cached; data blocks would also be
/// cached if there were a block_cache, but SplitCache has None for block_cache.
async fn warmup_meta_cache(db: &Db) {
    let opts = scan_options();
    // Scan a prefix that exists — this forces all SSTs to be opened (filter + index read).
    let mut iter = db.scan_prefix_with_options(b"px00", &opts).await.unwrap();
    while iter.next().await.unwrap().is_some() {}
}

fn scan_options() -> ScanOptions {
    ScanOptions {
        cache_blocks: true, // cache filter + index via meta_cache; no block_cache → data uncached
        durability_filter: DurabilityLevel::Remote, // skip memtables, read only from SSTs
        ..ScanOptions::default()
    }
}

/// Common settings: large SST size so each flush = 1 SST, no compaction.
fn base_settings() -> Settings {
    let mut s = Settings::default();
    s.min_filter_keys = 0;
    s.l0_sst_size_bytes = 256 * 1024 * 1024;
    s.l0_max_ssts = NUM_FLUSHES + 10;
    s.compactor_options = None;
    s
}

// ── Benchmark functions ─────────────────────────────────────────────────────

async fn bench_scan_prefix_first(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db.scan_prefix_with_options(QUERY_PREFIX, &opts).await.unwrap();
        let _entry = iter.next().await.unwrap();
        durations.push(start.elapsed());
    }
    durations
}

async fn bench_scan_prefix_by_recency_first(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let _entry = iter.next_entry().await.unwrap();
        durations.push(start.elapsed());
    }
    durations
}

async fn bench_scan_prefix_all(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db.scan_prefix_with_options(QUERY_PREFIX, &opts).await.unwrap();
        while iter.next().await.unwrap().is_some() {}
        durations.push(start.elapsed());
    }
    durations
}

async fn bench_scan_prefix_by_recency_all(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        while iter.next_entry().await.unwrap().is_some() {}
        durations.push(start.elapsed());
    }
    durations
}

// ── Statistics ───────────────────────────────────────────────────────────────

struct Stats {
    min: Duration,
    max: Duration,
    mean: Duration,
    p50: Duration,
    p99: Duration,
}

fn compute_stats(durations: &mut [Duration]) -> Stats {
    durations.sort();
    let n = durations.len();
    let sum: Duration = durations.iter().sum();
    Stats {
        min: durations[0],
        max: durations[n - 1],
        mean: sum / n as u32,
        p50: durations[n / 2],
        p99: durations[(n as f64 * 0.99) as usize],
    }
}

fn print_stats(label: &str, stats: &Stats) {
    println!(
        "  {:<45} min={:>8.2}ms  p50={:>8.2}ms  mean={:>8.2}ms  p99={:>8.2}ms  max={:>8.2}ms",
        label,
        stats.min.as_secs_f64() * 1e3,
        stats.p50.as_secs_f64() * 1e3,
        stats.mean.as_secs_f64() * 1e3,
        stats.p99.as_secs_f64() * 1e3,
        stats.max.as_secs_f64() * 1e3,
    );
}

// ── Gnuplot helpers ─────────────────────────────────────────────────────────

fn write_bar_chart_gnuplot(gnuplot_path: &str, png_path: &str, summary_csv: &str) {
    use std::io::Write;
    let h = "#";
    let mut f = std::fs::File::create(gnuplot_path).unwrap();
    writeln!(f, "set terminal pngcairo enhanced size 1200,600 font 'Helvetica,13'").unwrap();
    writeln!(f, "set output '{png_path}'").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "set style data histogram").unwrap();
    writeln!(f, "set style histogram clustered gap 1").unwrap();
    writeln!(f, "set style fill solid 0.8 border -1").unwrap();
    writeln!(f, "set boxwidth 0.9").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "set ylabel 'Latency (ms)' font 'Helvetica,14'").unwrap();
    writeln!(f, "set xlabel 'Strategy' font 'Helvetica,14'").unwrap();
    writeln!(
        f,
        "set title 'Prefix Scan Latency - p50 ({} L0 SSTs, {}ms GET, meta cached)' font 'Helvetica,15'",
        NUM_FLUSHES, GET_LATENCY.as_millis(),
    )
    .unwrap();
    writeln!(f).unwrap();
    writeln!(f, "set yrange [0:*]").unwrap();
    writeln!(f, "set grid ytics").unwrap();
    writeln!(f, "set key top left").unwrap();
    writeln!(f, "set xtics rotate by -15").unwrap();
    writeln!(f, "set datafile separator ','").unwrap();
    writeln!(f).unwrap();
    writeln!(
        f,
        "plot '{summary_csv}' every ::1 using 2:xtic(1) lc rgb '{h}2196F3' title 'First entry (p50)', '' every ::1 using 3 lc rgb '{h}FF9800' title 'All entries (p50)'"
    )
    .unwrap();
}

fn write_timeseries_gnuplot(gnuplot_path: &str, png_path: &str, csv_path: &str) {
    use std::io::Write;
    let h = "#";
    let mut f = std::fs::File::create(gnuplot_path).unwrap();
    writeln!(f, "set terminal pngcairo enhanced size 1400,500 font 'Helvetica,12'").unwrap();
    writeln!(f, "set output '{png_path}'").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "set ylabel 'Latency (ms)' font 'Helvetica,13'").unwrap();
    writeln!(f, "set xlabel 'Iteration' font 'Helvetica,13'").unwrap();
    writeln!(
        f,
        "set title 'Per-Iteration Latency: First Entry for Prefix ({} L0 SSTs, {}ms GET, meta cached)' font 'Helvetica,14'",
        NUM_FLUSHES, GET_LATENCY.as_millis(),
    )
    .unwrap();
    writeln!(f).unwrap();
    writeln!(f, "set yrange [0:*]").unwrap();
    writeln!(f, "set grid").unwrap();
    writeln!(f, "set key top right").unwrap();
    writeln!(f, "set datafile separator ','").unwrap();
    writeln!(f).unwrap();
    writeln!(
        f,
        "plot '{csv_path}' every ::1 using 1:2 with lines lw 1.5 lc rgb '{h}F44336' title 'No filter', '' every ::1 using 1:3 with lines lw 1.5 lc rgb '{h}2196F3' title 'Prefix bloom', '' every ::1 using 1:4 with lines lw 1.5 lc rgb '{h}4CAF50' title 'Recency iter'"
    )
    .unwrap();
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let rt = Runtime::new().unwrap();
    let total_keys = NUM_FLUSHES * PREFIXES_PER_FLUSH * KEYS_PER_PREFIX;
    println!("=== Prefix Scan Benchmark ===");
    println!(
        "  {} total prefixes, {} prefixes/flush x {} keys/prefix x {} flushes = {} total keys, {} L0 SSTs",
        NUM_PREFIXES, PREFIXES_PER_FLUSH, KEYS_PER_PREFIX, NUM_FLUSHES, total_keys, NUM_FLUSHES,
    );
    let ssts_with_prefix = NUM_FLUSHES * PREFIXES_PER_FLUSH / NUM_PREFIXES;
    println!(
        "  Query prefix {:?} appears in {} out of {} SSTs",
        std::str::from_utf8(QUERY_PREFIX).unwrap(),
        ssts_with_prefix,
        NUM_FLUSHES,
    );
    println!("  Durability filter: Remote (memtables skipped)");
    println!("  Meta cache (filter+index): ENABLED and warmed");
    println!("  Data block cache: DISABLED");
    println!("  Simulated GET latency: {}ms per call (data blocks only after warmup)", GET_LATENCY.as_millis());
    println!("  Iterations per measurement: {}", SCAN_ITERATIONS);
    println!();

    // ── 1. No prefix bloom filter (baseline) ────────────────────────────
    println!("--- Strategy 1: scan_prefix, NO prefix bloom filter ---");
    let store1 = make_throttled_store();
    let db_no_filter = rt.block_on(async {
        let db = Db::builder("/bench/no_prefix_filter", store1.clone())
            .with_settings(base_settings())
            .with_db_cache(meta_only_cache())
            .build()
            .await
            .unwrap();
        populate_db(&db).await;
        warmup_meta_cache(&db).await;
        db
    });
    enable_throttle(&store1);

    let mut d1_first = rt.block_on(bench_scan_prefix_first(&db_no_filter));
    let mut d1_all = rt.block_on(bench_scan_prefix_all(&db_no_filter));
    let s1_first = compute_stats(&mut d1_first);
    let s1_all = compute_stats(&mut d1_all);
    print_stats("scan_prefix (first entry)", &s1_first);
    print_stats("scan_prefix (all entries)", &s1_all);
    println!();

    // ── 2. With prefix bloom filter ─────────────────────────────────────
    println!("--- Strategy 2: scan_prefix, WITH prefix bloom filter ---");
    let store2 = make_throttled_store();
    let db_prefix_filter = rt.block_on(async {
        let mut settings = base_settings();
        settings.filter_policies = vec![Arc::new(
            BloomFilterPolicy::new(10)
                .with_prefix_extractor(Arc::new(FixedPrefixExtractor::new(4))),
        )];
        let db = Db::builder("/bench/prefix_filter", store2.clone())
            .with_settings(settings)
            .with_db_cache(meta_only_cache())
            .build()
            .await
            .unwrap();
        populate_db(&db).await;
        warmup_meta_cache(&db).await;
        db
    });
    enable_throttle(&store2);

    let mut d2_first = rt.block_on(bench_scan_prefix_first(&db_prefix_filter));
    let mut d2_all = rt.block_on(bench_scan_prefix_all(&db_prefix_filter));
    let s2_first = compute_stats(&mut d2_first);
    let s2_all = compute_stats(&mut d2_all);
    print_stats("scan_prefix (first entry)", &s2_first);
    print_stats("scan_prefix (all entries)", &s2_all);
    println!();

    // ── 3. Recency-based iterator with prefix bloom filter ──────────────
    println!("--- Strategy 3: scan_prefix_by_recency, WITH prefix bloom filter ---");
    let mut d3_first = rt.block_on(bench_scan_prefix_by_recency_first(&db_prefix_filter));
    let mut d3_all = rt.block_on(bench_scan_prefix_by_recency_all(&db_prefix_filter));
    let s3_first = compute_stats(&mut d3_first);
    let s3_all = compute_stats(&mut d3_all);
    print_stats("scan_prefix_by_recency (first entry)", &s3_first);
    print_stats("scan_prefix_by_recency (all entries)", &s3_all);
    println!();

    // ── Write CSV ────────────────────────────────────────────────────────
    let csv_path = "/tmp/scan_prefix_bench.csv";
    let gnuplot_path = "/tmp/scan_prefix_bench.gnuplot";
    let png_path = "/tmp/scan_prefix_bench.png";

    {
        use std::io::Write;
        let mut f = std::fs::File::create(csv_path).unwrap();
        writeln!(
            f,
            "iteration,no_filter_first_ms,prefix_filter_first_ms,recency_first_ms,no_filter_all_ms,prefix_filter_all_ms,recency_all_ms"
        )
        .unwrap();
        for i in 0..SCAN_ITERATIONS {
            writeln!(
                f,
                "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                i,
                d1_first[i].as_secs_f64() * 1e3,
                d2_first[i].as_secs_f64() * 1e3,
                d3_first[i].as_secs_f64() * 1e3,
                d1_all[i].as_secs_f64() * 1e3,
                d2_all[i].as_secs_f64() * 1e3,
                d3_all[i].as_secs_f64() * 1e3,
            )
            .unwrap();
        }
        println!("  CSV written to {csv_path}");
    }

    let summary_csv = "/tmp/scan_prefix_bench_summary.csv";
    {
        use std::io::Write;
        let mut f = std::fs::File::create(summary_csv).unwrap();
        writeln!(f, "strategy,first_entry_p50_ms,all_entries_p50_ms,first_entry_mean_ms,all_entries_mean_ms").unwrap();
        writeln!(
            f,
            "No prefix filter,{:.3},{:.3},{:.3},{:.3}",
            s1_first.p50.as_secs_f64() * 1e3, s1_all.p50.as_secs_f64() * 1e3,
            s1_first.mean.as_secs_f64() * 1e3, s1_all.mean.as_secs_f64() * 1e3,
        ).unwrap();
        writeln!(
            f,
            "Prefix bloom filter,{:.3},{:.3},{:.3},{:.3}",
            s2_first.p50.as_secs_f64() * 1e3, s2_all.p50.as_secs_f64() * 1e3,
            s2_first.mean.as_secs_f64() * 1e3, s2_all.mean.as_secs_f64() * 1e3,
        ).unwrap();
        writeln!(
            f,
            "Recency iterator,{:.3},{:.3},{:.3},{:.3}",
            s3_first.p50.as_secs_f64() * 1e3, s3_all.p50.as_secs_f64() * 1e3,
            s3_first.mean.as_secs_f64() * 1e3, s3_all.mean.as_secs_f64() * 1e3,
        ).unwrap();
    }

    // ── Gnuplot ──────────────────────────────────────────────────────────
    write_bar_chart_gnuplot(gnuplot_path, png_path, summary_csv);
    println!("  Gnuplot script written to {gnuplot_path}");

    match std::process::Command::new("gnuplot").arg(gnuplot_path).status() {
        Ok(s) if s.success() => println!("  Bar chart written to {png_path}"),
        Ok(s) => eprintln!("  gnuplot exited with {s}"),
        Err(e) => eprintln!("  gnuplot not found or failed: {e}"),
    }

    let ts_gnuplot = "/tmp/scan_prefix_bench_timeseries.gnuplot";
    let ts_png = "/tmp/scan_prefix_bench_timeseries.png";
    write_timeseries_gnuplot(ts_gnuplot, ts_png, csv_path);

    match std::process::Command::new("gnuplot").arg(ts_gnuplot).status() {
        Ok(s) if s.success() => println!("  Time-series chart written to {ts_png}"),
        Ok(s) => eprintln!("  gnuplot exited with {s}"),
        Err(e) => eprintln!("  gnuplot not found or failed: {e}"),
    }

    // Disable throttle before close to avoid slow shutdown
    disable_throttle(&store1);
    disable_throttle(&store2);
    rt.block_on(async {
        db_no_filter.close().await.unwrap();
        db_prefix_filter.close().await.unwrap();
    });

    println!("\nDone.");
}
