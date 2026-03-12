/// Benchmark comparing prefix scan strategies for a versioned-key workload:
///
///   Keys are "{prefix}:{version}" where prefix is the entity id and version
///   is a monotonically increasing u64 (old → new).  The benchmark inserts
///   versions across many flush rounds so that a single prefix's versions are
///   spread across multiple SSTs.
///
///   Strategies:
///     1. scan_prefix WITHOUT prefix bloom filter (baseline)
///     2. scan_prefix WITH prefix bloom filter
///     3. scan_prefix_by_recency WITH prefix bloom filter
///     4. scan_prefix_by_recency WITH prefix bloom filter + min/max version
///        filter (composite policies via the array-of-policies support),
///        passing a version_upper_bound hint
///
///   Each strategy is tested for:
///     - Reading the LATEST version of a key (first entry in recency order)
///     - Reading the OLDEST version of a key (requires full scan)
///
/// Run: cargo bench -p slatedb --bench scan_prefix_bench
///
/// Design:
/// - Meta cache (filter + index blocks) is enabled and warmed before measurement.
///   Data block cache is disabled so every data block read hits the throttled store.
/// - Object store wrapped with ThrottledStore (1ms per GET) to simulate latency.
/// - 100 prefixes, each getting versions across 100 flush rounds.
/// - Sentinel keys in every flush ensure all SSTs cover the full key range.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::{BufMut, Bytes};
use object_store::memory::InMemory;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use slatedb::config::{
    DurabilityLevel, FlushOptions, FlushType, PutOptions, ScanOptions, Settings, WriteOptions,
};
use slatedb::db_cache::SplitCache;
use slatedb::{
    BloomFilterPolicy, Db, Filter, FilterBuilder, FilterPolicy, FilterQuery, FixedPrefixExtractor,
};
use tokio::runtime::Runtime;

// ── Parameters ──────────────────────────────────────────────────────────────

/// Total distinct prefixes ("pk0000".."pk0099").
const NUM_PREFIXES: usize = 100;
/// Prefixes written per flush round (round-robin from the 100).
/// With 10 prefixes per flush and 100 rounds, each prefix appears in 10 SSTs.
const PREFIXES_PER_FLUSH: usize = 10;
/// Versions per prefix within a single flush round.
const VERSIONS_PER_FLUSH: usize = 5;
/// Number of flush rounds → L0 SSTs.
const NUM_FLUSHES: usize = 100;
/// Scan iterations for stable averages.
const SCAN_ITERATIONS: usize = 20;
/// Prefix length in bytes ("pk0000" = 6 bytes).
const PREFIX_LEN: usize = 6;
/// Query prefix that appears in 10 SSTs.
const QUERY_PREFIX: &[u8; 6] = b"pk0095";
/// Total versions for any given prefix = (NUM_FLUSHES / (NUM_PREFIXES / PREFIXES_PER_FLUSH)) * VERSIONS_PER_FLUSH
/// = (100 / 10) * 5 = 50 versions per prefix.
/// Simulated GET latency per call.
const GET_LATENCY: Duration = Duration::from_millis(1);
/// Hint key for the version upper bound used by MinMaxVersionFilter.
const VERSION_UPPER_BOUND_HINT: &str = "version_upper_bound";

// ── Key encoding ────────────────────────────────────────────────────────────

/// Key format: "{prefix}:{version:08}" e.g. "pk0042:00000017"
fn make_key(prefix_idx: usize, version: u64) -> Vec<u8> {
    format!("pk{:04}:{:08}", prefix_idx, version).into_bytes()
}

fn make_value(round: usize, v: usize) -> Vec<u8> {
    format!("val-{}-{}", round, v).into_bytes()
}

/// Extract the version number from a key like "pk0042:00000017"
fn parse_version_from_key(key: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(key).ok()?;
    let colon = s.find(':')?;
    s[colon + 1..].parse().ok()
}

// ── MinMaxVersionFilter ─────────────────────────────────────────────────────
//
// A custom filter policy that tracks the min and max version (the u64 after
// the ':' in each key) per SST.  During reads, if the query carries a
// `version_upper_bound` hint, the filter can skip SSTs whose min version
// exceeds the bound.

struct MinMaxVersionFilter {
    min_version: u64,
    max_version: u64,
}

impl Filter for MinMaxVersionFilter {
    fn might_match(&self, query: &FilterQuery) -> bool {
        // If the caller provided a version upper bound hint, skip the SST
        // if all versions in it are above the bound.
        if let Some(bound_bytes) = query.hints.get(VERSION_UPPER_BOUND_HINT) {
            if let Ok(s) = std::str::from_utf8(bound_bytes.as_ref()) {
                if let Ok(upper) = s.parse::<u64>() {
                    // SST's min_version > upper bound → no matching version here
                    if self.min_version > upper {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn encode(&self, writer: &mut dyn BufMut) {
        writer.put_u64(self.min_version);
        writer.put_u64(self.max_version);
    }

    fn size(&self) -> usize {
        16 // two u64s
    }
}

struct MinMaxVersionFilterBuilder {
    min_version: u64,
    max_version: u64,
    has_keys: bool,
}

impl FilterBuilder for MinMaxVersionFilterBuilder {
    fn add_key(&mut self, key: &[u8]) {
        if let Some(v) = parse_version_from_key(key) {
            if !self.has_keys {
                self.min_version = v;
                self.max_version = v;
                self.has_keys = true;
            } else {
                self.min_version = self.min_version.min(v);
                self.max_version = self.max_version.max(v);
            }
        }
    }

    fn build(&self) -> Arc<dyn Filter> {
        Arc::new(MinMaxVersionFilter {
            min_version: self.min_version,
            max_version: self.max_version,
        })
    }
}

struct MinMaxVersionFilterPolicy;

impl FilterPolicy for MinMaxVersionFilterPolicy {
    fn name(&self) -> &str {
        "bench.MinMaxVersion"
    }

    fn builder(&self) -> Box<dyn FilterBuilder> {
        Box::new(MinMaxVersionFilterBuilder {
            min_version: u64::MAX,
            max_version: 0,
            has_keys: false,
        })
    }

    fn decode(&self, data: &[u8]) -> Arc<dyn Filter> {
        let min_version = u64::from_be_bytes(data[0..8].try_into().unwrap());
        let max_version = u64::from_be_bytes(data[8..16].try_into().unwrap());
        Arc::new(MinMaxVersionFilter {
            min_version,
            max_version,
        })
    }

    fn estimate_size(&self, _num_keys: usize) -> usize {
        16
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_throttled_store() -> Arc<ThrottledStore<InMemory>> {
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

fn meta_only_cache() -> Arc<SplitCache> {
    use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
    let meta = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
        max_capacity: 256 * 1024 * 1024,
        ..Default::default()
    }));
    Arc::new(
        SplitCache::new()
            .with_block_cache(None)
            .with_meta_cache(Some(meta))
            .build(),
    )
}

/// Populate the DB with versioned keys.
///
/// Each flush round writes PREFIXES_PER_FLUSH prefixes (round-robin).
/// Within each flush, each prefix gets VERSIONS_PER_FLUSH new versions.
/// Versions are globally increasing: round 0 gets versions 0..4,
/// round 1 gets 5..9, etc.  So older SSTs contain lower version numbers.
///
/// Sentinel keys at the start ("aa_*") and end ("zz_*") of the key space
/// ensure every SST's key range overlaps with every prefix range.
async fn populate_db(db: &Db) {
    let write_opts = WriteOptions {
        await_durable: false,
    };
    let put_opts = PutOptions::default();

    // Track how many versions each prefix has gotten so far
    let mut next_version = vec![0u64; NUM_PREFIXES];

    for round in 0..NUM_FLUSHES {
        // Sentinel keys
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
            for _ in 0..VERSIONS_PER_FLUSH {
                let version = next_version[px];
                next_version[px] += 1;
                let key = make_key(px, version);
                let val = make_value(round, version as usize);
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

async fn warmup_meta_cache(db: &Db) {
    let opts = scan_options();
    let mut iter = db
        .scan_prefix_with_options(b"pk0000", &opts)
        .await
        .unwrap();
    while iter.next().await.unwrap().is_some() {}
}

fn scan_options() -> ScanOptions {
    ScanOptions {
        cache_blocks: true,
        durability_filter: DurabilityLevel::Remote,
        ..ScanOptions::default()
    }
}

fn scan_options_with_version_hint(upper_bound: u64) -> ScanOptions {
    let mut opts = scan_options();
    opts.filter_hints = HashMap::from([(
        VERSION_UPPER_BOUND_HINT.to_string(),
        Bytes::from(upper_bound.to_string()),
    )]);
    opts
}

fn base_settings() -> Settings {
    let mut s = Settings::default();
    s.min_filter_keys = 0;
    s.l0_sst_size_bytes = 256 * 1024 * 1024;
    s.l0_max_ssts = NUM_FLUSHES + 10;
    s.compactor_options = None;
    s
}

// ── Benchmark functions ─────────────────────────────────────────────────────

/// Read the latest version: scan_prefix returns keys in ascending order,
/// so we must read all entries and take the last one.
async fn bench_scan_prefix_latest(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let mut last = None;
        while let Some(entry) = iter.next().await.unwrap() {
            last = Some(entry);
        }
        assert!(last.is_some(), "expected at least one entry");
        durations.push(start.elapsed());
    }
    durations
}

/// Read the oldest version: scan_prefix returns ascending, first entry is oldest.
async fn bench_scan_prefix_oldest(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let entry = iter.next().await.unwrap();
        assert!(entry.is_some(), "expected at least one entry");
        durations.push(start.elapsed());
    }
    durations
}

/// Read latest version via recency iterator (first entry = most recent).
async fn bench_recency_latest(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let entry = iter.next_entry().await.unwrap();
        assert!(entry.is_some(), "expected at least one entry");
        durations.push(start.elapsed());
    }
    durations
}

/// Read oldest version via recency iterator (must read all entries).
async fn bench_recency_oldest(db: &Db) -> Vec<Duration> {
    let opts = scan_options();
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let mut last = None;
        while let Some(entry) = iter.next_entry().await.unwrap() {
            last = Some(entry);
        }
        assert!(last.is_some(), "expected at least one entry");
        durations.push(start.elapsed());
    }
    durations
}

/// Read latest version via recency iterator + version hint.
/// The hint tells the min/max filter "I only want versions <= upper_bound",
/// so SSTs whose min_version > upper_bound are skipped.
async fn bench_recency_latest_with_hint(db: &Db, upper_bound: u64) -> Vec<Duration> {
    let opts = scan_options_with_version_hint(upper_bound);
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let entry = iter.next_entry().await.unwrap();
        assert!(entry.is_some(), "expected at least one entry");
        durations.push(start.elapsed());
    }
    durations
}

/// Read oldest version via recency iterator + version hint targeting oldest version.
async fn bench_recency_oldest_with_hint(db: &Db, upper_bound: u64) -> Vec<Duration> {
    let opts = scan_options_with_version_hint(upper_bound);
    let mut durations = Vec::with_capacity(SCAN_ITERATIONS);
    for _ in 0..SCAN_ITERATIONS {
        let start = Instant::now();
        let mut iter = db
            .scan_prefix_by_recency_with_options(QUERY_PREFIX, &opts)
            .await
            .unwrap();
        let mut last = None;
        while let Some(entry) = iter.next_entry().await.unwrap() {
            last = Some(entry);
        }
        assert!(last.is_some(), "expected at least one entry");
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
        "  {:<55} min={:>8.2}ms  p50={:>8.2}ms  mean={:>8.2}ms  p99={:>8.2}ms  max={:>8.2}ms",
        label,
        stats.min.as_secs_f64() * 1e3,
        stats.p50.as_secs_f64() * 1e3,
        stats.mean.as_secs_f64() * 1e3,
        stats.p99.as_secs_f64() * 1e3,
        stats.max.as_secs_f64() * 1e3,
    );
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let rt = Runtime::new().unwrap();

    let versions_per_prefix =
        (NUM_FLUSHES * PREFIXES_PER_FLUSH / NUM_PREFIXES) * VERSIONS_PER_FLUSH;
    let total_keys = NUM_FLUSHES * PREFIXES_PER_FLUSH * VERSIONS_PER_FLUSH;
    let ssts_with_prefix = NUM_FLUSHES * PREFIXES_PER_FLUSH / NUM_PREFIXES;
    let max_version = versions_per_prefix as u64 - 1;

    println!("=== Versioned-Key Prefix Scan Benchmark ===");
    println!(
        "  {} prefixes, {} versions/prefix, {} total keys, {} L0 SSTs",
        NUM_PREFIXES, versions_per_prefix, total_keys, NUM_FLUSHES,
    );
    println!(
        "  Query prefix {:?} appears in {} out of {} SSTs, versions 0..{}",
        std::str::from_utf8(QUERY_PREFIX).unwrap(),
        ssts_with_prefix,
        NUM_FLUSHES,
        max_version,
    );
    println!("  Durability filter: Remote (memtables skipped)");
    println!("  Meta cache: ENABLED and warmed | Data block cache: DISABLED");
    println!(
        "  Simulated GET latency: {}ms | Iterations: {}",
        GET_LATENCY.as_millis(),
        SCAN_ITERATIONS
    );
    println!();

    // ── 1. No prefix bloom filter (baseline) ────────────────────────────
    println!("--- Strategy 1: scan_prefix, NO prefix bloom filter ---");
    let store1 = make_throttled_store();
    let db_no_filter = rt.block_on(async {
        let db = Db::builder("/bench/no_filter", store1.clone())
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

    let mut d1_latest = rt.block_on(bench_scan_prefix_latest(&db_no_filter));
    let mut d1_oldest = rt.block_on(bench_scan_prefix_oldest(&db_no_filter));
    let s1_latest = compute_stats(&mut d1_latest);
    let s1_oldest = compute_stats(&mut d1_oldest);
    print_stats("scan_prefix (latest version = full scan)", &s1_latest);
    print_stats("scan_prefix (oldest version = first entry)", &s1_oldest);
    println!();

    // ── 2. With prefix bloom filter ─────────────────────────────────────
    println!("--- Strategy 2: scan_prefix, WITH prefix bloom filter ---");
    let store2 = make_throttled_store();
    let db_prefix_bloom = rt.block_on(async {
        let mut settings = base_settings();
        settings.filter_policies = vec![Arc::new(
            BloomFilterPolicy::new(10)
                .with_prefix_extractor(Arc::new(FixedPrefixExtractor::new(PREFIX_LEN))),
        )];
        let db = Db::builder("/bench/prefix_bloom", store2.clone())
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

    let mut d2_latest = rt.block_on(bench_scan_prefix_latest(&db_prefix_bloom));
    let mut d2_oldest = rt.block_on(bench_scan_prefix_oldest(&db_prefix_bloom));
    let s2_latest = compute_stats(&mut d2_latest);
    let s2_oldest = compute_stats(&mut d2_oldest);
    print_stats("scan_prefix (latest version = full scan)", &s2_latest);
    print_stats("scan_prefix (oldest version = first entry)", &s2_oldest);
    println!();

    // ── 3. Recency iterator + prefix bloom filter ───────────────────────
    println!("--- Strategy 3: recency iterator, WITH prefix bloom filter ---");
    let mut d3_latest = rt.block_on(bench_recency_latest(&db_prefix_bloom));
    let mut d3_oldest = rt.block_on(bench_recency_oldest(&db_prefix_bloom));
    let s3_latest = compute_stats(&mut d3_latest);
    let s3_oldest = compute_stats(&mut d3_oldest);
    print_stats("recency (latest version = first entry)", &s3_latest);
    print_stats("recency (oldest version = full scan)", &s3_oldest);
    println!();

    // ── 4. Recency iterator + prefix bloom + min/max version filter ─────
    println!("--- Strategy 4: recency + prefix bloom + min/max version filter ---");
    let store4 = make_throttled_store();
    let db_composite = rt.block_on(async {
        let mut settings = base_settings();
        settings.filter_policies = vec![
            Arc::new(
                BloomFilterPolicy::new(10)
                    .with_prefix_extractor(Arc::new(FixedPrefixExtractor::new(PREFIX_LEN))),
            ),
            Arc::new(MinMaxVersionFilterPolicy),
        ];
        let db = Db::builder("/bench/composite_filter", store4.clone())
            .with_settings(settings)
            .with_db_cache(meta_only_cache())
            .build()
            .await
            .unwrap();
        populate_db(&db).await;
        warmup_meta_cache(&db).await;
        db
    });
    enable_throttle(&store4);

    // Latest version: hint = max_version (all SSTs pass min/max check)
    let mut d4_latest = rt.block_on(bench_recency_latest_with_hint(&db_composite, max_version));
    // Oldest version: hint = 4 (only SSTs with min_version <= 4 survive)
    let mut d4_oldest =
        rt.block_on(bench_recency_oldest_with_hint(&db_composite, VERSIONS_PER_FLUSH as u64 - 1));
    let s4_latest = compute_stats(&mut d4_latest);
    let s4_oldest = compute_stats(&mut d4_oldest);
    print_stats("recency (latest version, hint=max)", &s4_latest);
    print_stats("recency (oldest version, hint=4)", &s4_oldest);
    println!();

    // ── Write CSV ────────────────────────────────────────────────────────
    let csv_path = "/tmp/scan_prefix_bench.csv";
    {
        use std::io::Write;
        let mut f = std::fs::File::create(csv_path).unwrap();
        writeln!(
            f,
            "iteration,no_filter_latest_ms,no_filter_oldest_ms,bloom_latest_ms,bloom_oldest_ms,recency_latest_ms,recency_oldest_ms,composite_latest_ms,composite_oldest_ms"
        )
        .unwrap();
        for i in 0..SCAN_ITERATIONS {
            writeln!(
                f,
                "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                i,
                d1_latest[i].as_secs_f64() * 1e3,
                d1_oldest[i].as_secs_f64() * 1e3,
                d2_latest[i].as_secs_f64() * 1e3,
                d2_oldest[i].as_secs_f64() * 1e3,
                d3_latest[i].as_secs_f64() * 1e3,
                d3_oldest[i].as_secs_f64() * 1e3,
                d4_latest[i].as_secs_f64() * 1e3,
                d4_oldest[i].as_secs_f64() * 1e3,
            )
            .unwrap();
        }
        println!("  CSV written to {csv_path}");
    }

    let summary_csv = "/tmp/scan_prefix_bench_summary.csv";
    {
        use std::io::Write;
        let mut f = std::fs::File::create(summary_csv).unwrap();
        writeln!(f, "strategy,latest_p50_ms,oldest_p50_ms,latest_mean_ms,oldest_mean_ms").unwrap();
        writeln!(
            f,
            "No filter,{:.3},{:.3},{:.3},{:.3}",
            s1_latest.p50.as_secs_f64() * 1e3,
            s1_oldest.p50.as_secs_f64() * 1e3,
            s1_latest.mean.as_secs_f64() * 1e3,
            s1_oldest.mean.as_secs_f64() * 1e3,
        )
        .unwrap();
        writeln!(
            f,
            "Prefix bloom,{:.3},{:.3},{:.3},{:.3}",
            s2_latest.p50.as_secs_f64() * 1e3,
            s2_oldest.p50.as_secs_f64() * 1e3,
            s2_latest.mean.as_secs_f64() * 1e3,
            s2_oldest.mean.as_secs_f64() * 1e3,
        )
        .unwrap();
        writeln!(
            f,
            "Recency + bloom,{:.3},{:.3},{:.3},{:.3}",
            s3_latest.p50.as_secs_f64() * 1e3,
            s3_oldest.p50.as_secs_f64() * 1e3,
            s3_latest.mean.as_secs_f64() * 1e3,
            s3_oldest.mean.as_secs_f64() * 1e3,
        )
        .unwrap();
        writeln!(
            f,
            "Recency + bloom + minmax,{:.3},{:.3},{:.3},{:.3}",
            s4_latest.p50.as_secs_f64() * 1e3,
            s4_oldest.p50.as_secs_f64() * 1e3,
            s4_latest.mean.as_secs_f64() * 1e3,
            s4_oldest.mean.as_secs_f64() * 1e3,
        )
        .unwrap();
        println!("  Summary CSV written to {summary_csv}");
    }

    // ── Gnuplot charts ─────────────────────────────────────────────────
    let bar_gp = "/tmp/scan_prefix_bench_bar.gnuplot";
    let bar_png = "/tmp/scan_prefix_bench_bar.png";
    write_bar_chart_gnuplot(bar_gp, bar_png, summary_csv);
    run_gnuplot(bar_gp, bar_png);

    let ts_gp = "/tmp/scan_prefix_bench_ts.gnuplot";
    let ts_png = "/tmp/scan_prefix_bench_ts.png";
    write_timeseries_gnuplot(ts_gp, ts_png, csv_path);
    run_gnuplot(ts_gp, ts_png);

    // Disable throttle before close
    disable_throttle(&store1);
    disable_throttle(&store2);
    disable_throttle(&store4);
    rt.block_on(async {
        db_no_filter.close().await.unwrap();
        db_prefix_bloom.close().await.unwrap();
        db_composite.close().await.unwrap();
    });

    println!("\nDone.");
}

// ── Gnuplot ─────────────────────────────────────────────────────────────────

fn run_gnuplot(script: &str, output: &str) {
    match std::process::Command::new("gnuplot").arg(script).status() {
        Ok(s) if s.success() => println!("  Chart written to {output}"),
        Ok(s) => eprintln!("  gnuplot exited with {s}"),
        Err(e) => eprintln!("  gnuplot not found or failed: {e}"),
    }
}

/// Grouped bar chart: p50 latency for "latest version" and "oldest version"
/// across all strategies.  Style modeled on the SlateDB benchmark charts in
/// https://github.com/slatedb/slatedb/issues/1302.
fn write_bar_chart_gnuplot(gp_path: &str, png_path: &str, summary_csv: &str) {
    use std::io::Write;
    let h = "#"; // avoids gnuplot interpreting '#' in format strings
    let mut f = std::fs::File::create(gp_path).unwrap();
    write!(
        f,
        r#"set terminal pngcairo enhanced size 1200,600 font 'Helvetica,12'
set output '{png_path}'

set style data histogram
set style histogram clustered gap 1
set style fill solid 0.8 border -1
set boxwidth 0.9

set ylabel 'Latency (ms)' font 'Helvetica,13'
set title 'Versioned Prefix Scan — p50 Latency ({num_ssts} L0 SSTs, {get_ms}ms GET, meta cached)' font 'Helvetica,14'

set yrange [0:*]
set grid ytics
set key top left font 'Helvetica,11'
set xtics rotate by -20 font 'Helvetica,10'
set datafile separator ','

plot '{summary_csv}' every ::1 using 2:xtic(1) lc rgb '{h}2196F3' title 'Latest version (p50)', \
     '' every ::1 using 3 lc rgb '{h}FF9800' title 'Oldest version (p50)'
"#,
        png_path = png_path,
        summary_csv = summary_csv,
        num_ssts = NUM_FLUSHES,
        get_ms = GET_LATENCY.as_millis(),
        h = h,
    )
    .unwrap();
    println!("  Gnuplot bar chart script written to {gp_path}");
}

/// Per-iteration time-series chart showing latency for each strategy's
/// "latest version" query.  Uses points + lines like the charts in issue #1302.
fn write_timeseries_gnuplot(gp_path: &str, png_path: &str, csv_path: &str) {
    use std::io::Write;
    let h = "#";
    let mut f = std::fs::File::create(gp_path).unwrap();
    write!(
        f,
        r#"set terminal pngcairo enhanced size 1400,500 font 'Helvetica,12'
set output '{png_path}'

set ylabel 'Latency (ms)' font 'Helvetica,13'
set xlabel 'Iteration' font 'Helvetica,13'
set title 'Per-Iteration Latency — Latest Version ({num_ssts} L0 SSTs, {get_ms}ms GET, meta cached)' font 'Helvetica,14'

set yrange [0:*]
set grid
set key top right font 'Helvetica,10'
set datafile separator ','

plot '{csv_path}' every ::1 using 1:2 with linespoints lw 1.5 pt 7 ps 0.6 lc rgb '{h}4CAF50' title 'No filter', \
     '' every ::1 using 1:4 with linespoints lw 1.5 pt 7 ps 0.6 lc rgb '{h}2196F3' title 'Prefix bloom', \
     '' every ::1 using 1:6 with linespoints lw 1.5 pt 7 ps 0.6 lc rgb '{h}FF9800' title 'Recency + bloom', \
     '' every ::1 using 1:8 with linespoints lw 1.5 pt 7 ps 0.6 lc rgb '{h}F44336' title 'Recency + bloom + minmax'
"#,
        png_path = png_path,
        csv_path = csv_path,
        num_ssts = NUM_FLUSHES,
        get_ms = GET_LATENCY.as_millis(),
        h = h,
    )
    .unwrap();
    println!("  Gnuplot time-series script written to {gp_path}");
}
