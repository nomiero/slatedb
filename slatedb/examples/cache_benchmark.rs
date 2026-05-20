// Heavy read-write benchmark for the on-disk object cache.
//
// Compares two configurations against the same workload:
//
//   baseline: default cache options (evictor on at 16GB, 4MB parts,
//             prewarm OFF, preload OFF, cache_puts OFF, fsync ON)
//
//   tuned:    evictor OFF (max_cache_size_bytes = None), 256MB parts,
//             prewarm ON, preload AllSst on startup, cache_puts ON,
//             fsync removed (in code), scc-backed fd cache, fadvise
//             RANDOM (in code).
//
// Run with:
//   cargo run --release --example cache_benchmark -- baseline
//   cargo run --release --example cache_benchmark -- tuned
//
// Upstream is a LocalFileSystem so that we exercise the full cache hot
// path without needing a real cloud bucket. The cache lives at a
// different directory.

use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use slatedb::config::{ObjectStoreCacheOptions, PreloadLevel, Settings, WriteOptions};
use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
use slatedb::db_cache::{DbCache, SplitCache};
use slatedb::Db;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const KEY_LEN: usize = 16;
const VAL_LEN: usize = 1024;
const KEY_COUNT: u64 = 50_000;
const WARMUP_CONCURRENCY: usize = 32;
const PUT_PERCENTAGE: u32 = 5;
const RUN_CONCURRENCY: usize = 32;
const RUN_OPS_PER_TASK: u64 = 5_000;

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tuned".to_string());

    let tmp_root = std::env::temp_dir().join(format!("slatedb_cache_bench_{}", mode));
    let upstream_dir = tmp_root.join("upstream");
    let cache_dir = tmp_root.join("cache");
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&upstream_dir).unwrap();
    std::fs::create_dir_all(&cache_dir).unwrap();

    // Simulate a remote upstream by adding a per-request delay. With both
    // the cache and upstream on the same local disk, no per-request delay
    // means the cache adds pure overhead with no win to offset it. 5ms is
    // a generous lower bound for S3 round-trip on the same region.
    let raw: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&upstream_dir).expect("create local upstream"));
    let upstream: Arc<dyn ObjectStore> = Arc::new(SlowObjectStore {
        inner: raw,
        delay: std::time::Duration::from_millis(20),
    });

    let mut settings = Settings::default();
    // Force the memtable to flush often so the workload actually spills into
    // SSTs (and therefore exercises the on-disk object cache). With the
    // default 512MB the entire 50MB warmup would sit in memory.
    settings.max_unflushed_bytes = 4 * 1024 * 1024;
    settings.l0_sst_size_bytes = 4 * 1024 * 1024;
    match mode.as_str() {
        "baseline" => {
            settings.object_store_cache_options = ObjectStoreCacheOptions {
                root_folder: Some(cache_dir.clone()),
                ..ObjectStoreCacheOptions::default()
            };
            // Even baseline gets prewarm off, preload off explicitly so the
            // contrast is honest:
            settings
                .object_store_cache_options
                .prewarm_cache_on_compaction = false;
            settings
                .object_store_cache_options
                .preload_disk_cache_on_startup = None;
            settings.object_store_cache_options.cache_puts = false;
        }
        "tuned" => {
            settings.object_store_cache_options = ObjectStoreCacheOptions {
                root_folder: Some(cache_dir.clone()),
                max_cache_size_bytes: None,
                part_size_bytes: 256 * 1024 * 1024,
                cache_puts: true,
                preload_disk_cache_on_startup: Some(PreloadLevel::AllSst),
                scan_interval: None,
                max_open_file_handles: 4096,
                prewarm_cache_on_compaction: true,
                use_io_uring: false,
                direct_io: false,
            };
        }
        other => {
            eprintln!("unknown mode: {} (expected baseline or tuned)", other);
            std::process::exit(2);
        }
    }

    println!("=== mode: {} ===", mode);
    println!("upstream: {}", upstream_dir.display());
    println!("cache:    {}", cache_dir.display());
    println!(
        "config:   part_size={}MB, evictor={}, prewarm={}, preload={:?}, cache_puts={}",
        settings.object_store_cache_options.part_size_bytes / (1024 * 1024),
        settings
            .object_store_cache_options
            .max_cache_size_bytes
            .map(|b| format!("on@{}MB", b / (1024 * 1024)))
            .unwrap_or_else(|| "OFF".to_string()),
        settings
            .object_store_cache_options
            .prewarm_cache_on_compaction,
        settings
            .object_store_cache_options
            .preload_disk_cache_on_startup,
        settings.object_store_cache_options.cache_puts,
    );

    // Per the workload spec, only filters and indexes are cached in memory.
    // Data blocks must miss the in-memory cache and go to the on-disk cache,
    // which is what we are actually measuring. Both modes use the same
    // memory-cache shape so the comparison is honest.
    let make_split_cache = || {
        let meta_cache: Arc<dyn DbCache> = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
            max_capacity: 256 * 1024 * 1024,
            ..Default::default()
        }));
        Arc::new(
            SplitCache::new()
                .with_block_cache(None)
                .with_meta_cache(Some(meta_cache))
                .build(),
        ) as Arc<dyn DbCache>
    };

    let settings_for_reopen = settings.clone();
    let db_path = Path::from("bench-db");
    let db = Arc::new(
        Db::builder(db_path.clone(), upstream.clone())
            .with_settings(settings)
            .with_db_cache(make_split_cache())
            .build()
            .await
            .expect("open db"),
    );

    // Pre-generate the key set.
    let keys = generate_keys(KEY_COUNT, KEY_LEN);

    // Warmup: write the full key set with concurrent non-durable puts so we
    // saturate the WAL pipeline instead of one-shot blocking on each put.
    println!(
        "warmup: writing {} keys ({} concurrent)",
        KEY_COUNT, WARMUP_CONCURRENCY
    );
    let warmup_start = Instant::now();
    let chunks_per_task = KEY_COUNT.div_ceil(WARMUP_CONCURRENCY as u64) as usize;
    let mut warmup_handles = Vec::with_capacity(WARMUP_CONCURRENCY);
    for tid in 0..WARMUP_CONCURRENCY {
        let db = db.clone();
        let keys = keys.clone();
        let lo = (tid * chunks_per_task).min(keys.len());
        let hi = ((tid + 1) * chunks_per_task).min(keys.len());
        warmup_handles.push(tokio::spawn(async move {
            let opts = WriteOptions {
                await_durable: false,
                #[cfg(dst)]
                now: 0,
            };
            for key in &keys[lo..hi] {
                let val = vec![0u8; VAL_LEN];
                db.put_with_options(
                    key.as_slice(),
                    &val[..],
                    &slatedb::config::PutOptions::default(),
                    &opts,
                )
                .await
                .expect("warmup put");
            }
        }));
    }
    for h in warmup_handles {
        h.await.unwrap();
    }
    db.flush().await.expect("flush warmup");
    println!("warmup done in {:?}", warmup_start.elapsed());

    // Wait for L0 SSTs to settle. With l0_sst_size_bytes=4MB and a 50MB
    // workload we expect ~12 SSTs to land before this sleep returns. The run
    // phase will read mostly through the SSTs, exercising the cache.
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    db.flush().await.expect("post-warmup flush");

    // Close + reopen the DB to drop all in-memory state. This ensures the
    // run phase starts with a cold memtable and a cold in-memory meta
    // cache, so reads must traverse the on-disk cache (or upstream on
    // miss). For the tuned config, the preload step at re-open should
    // pre-populate the on-disk cache; for baseline, the on-disk cache is
    // empty and reads fault through.
    println!("re-opening db to drop in-memory state");
    db.close().await.expect("close db");
    drop(db);
    let reopen_start = Instant::now();
    let db = Arc::new(
        Db::builder(db_path.clone(), upstream.clone())
            .with_settings(settings_for_reopen.clone())
            .with_db_cache(make_split_cache())
            .build()
            .await
            .expect("reopen db"),
    );
    println!("reopened in {:?}", reopen_start.elapsed());

    // Report on-disk cache state after reopen + preload.
    let cache_files = walkdir::WalkDir::new(&cache_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .count();
    let cache_bytes: u64 = walkdir::WalkDir::new(&cache_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    let upstream_files = walkdir::WalkDir::new(&upstream_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .count();
    println!(
        "cache state: {} files, {:.1} MB on disk; upstream: {} files",
        cache_files,
        cache_bytes as f64 / (1024.0 * 1024.0),
        upstream_files,
    );

    println!(
        "run: {} tasks * {} ops, put_pct={}",
        RUN_CONCURRENCY, RUN_OPS_PER_TASK, PUT_PERCENTAGE
    );
    let run_start = Instant::now();

    let total_ops = Arc::new(AtomicU64::new(0));
    let get_count = Arc::new(AtomicU64::new(0));
    let put_count = Arc::new(AtomicU64::new(0));
    let get_lat_ns = Arc::new(AtomicU64::new(0));
    let put_lat_ns = Arc::new(AtomicU64::new(0));
    let get_samples = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(100_000)));
    let put_samples = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(50_000)));

    let mut handles = Vec::with_capacity(RUN_CONCURRENCY);
    for task_id in 0..RUN_CONCURRENCY {
        let db = db.clone();
        let keys = keys.clone();
        let total_ops = total_ops.clone();
        let get_count = get_count.clone();
        let put_count = put_count.clone();
        let get_lat_ns = get_lat_ns.clone();
        let put_lat_ns = put_lat_ns.clone();
        let get_samples = get_samples.clone();
        let put_samples = put_samples.clone();
        handles.push(tokio::spawn(async move {
            let mut rng =
                rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xdead_beef + task_id as u64);
            let mut local_gets = Vec::with_capacity(RUN_OPS_PER_TASK as usize);
            let mut local_puts = Vec::with_capacity(RUN_OPS_PER_TASK as usize / 2);
            let put_opts = WriteOptions {
                await_durable: false,
                #[cfg(dst)]
                now: 0,
            };
            for _ in 0..RUN_OPS_PER_TASK {
                let is_put = rng.random_range(0..100) < PUT_PERCENTAGE;
                let key_idx = rng.random_range(0..keys.len());
                let key = &keys[key_idx];
                if is_put {
                    let val = vec![0xab_u8; VAL_LEN];
                    let t0 = Instant::now();
                    db.put_with_options(
                        key.as_slice(),
                        &val[..],
                        &slatedb::config::PutOptions::default(),
                        &put_opts,
                    )
                    .await
                    .expect("put");
                    let dt = t0.elapsed().as_nanos() as u64;
                    put_lat_ns.fetch_add(dt, Ordering::Relaxed);
                    put_count.fetch_add(1, Ordering::Relaxed);
                    local_puts.push(dt);
                } else {
                    let t0 = Instant::now();
                    let _ = db.get(key.as_slice()).await.expect("get");
                    let dt = t0.elapsed().as_nanos() as u64;
                    get_lat_ns.fetch_add(dt, Ordering::Relaxed);
                    get_count.fetch_add(1, Ordering::Relaxed);
                    local_gets.push(dt);
                }
                total_ops.fetch_add(1, Ordering::Relaxed);
            }
            get_samples.lock().extend(local_gets);
            put_samples.lock().extend(local_puts);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let elapsed = run_start.elapsed();

    let total = total_ops.load(Ordering::Relaxed);
    let gets = get_count.load(Ordering::Relaxed);
    let puts = put_count.load(Ordering::Relaxed);
    let avg_get_us = if gets > 0 {
        get_lat_ns.load(Ordering::Relaxed) as f64 / (gets as f64 * 1000.0)
    } else {
        0.0
    };
    let avg_put_us = if puts > 0 {
        put_lat_ns.load(Ordering::Relaxed) as f64 / (puts as f64 * 1000.0)
    } else {
        0.0
    };
    let mut g = get_samples.lock();
    g.sort_unstable();
    let g50 = pct(&g, 0.50);
    let g95 = pct(&g, 0.95);
    let g99 = pct(&g, 0.99);
    let mut p = put_samples.lock();
    p.sort_unstable();
    let p50 = pct(&p, 0.50);
    let p95 = pct(&p, 0.95);
    let p99 = pct(&p, 0.99);
    let throughput = total as f64 / elapsed.as_secs_f64();

    println!();
    println!("=== results ({}) ===", mode);
    println!("elapsed:     {:?}", elapsed);
    println!("total ops:   {} ({} gets, {} puts)", total, gets, puts);
    println!("throughput:  {:.0} ops/s", throughput);
    println!("avg get:     {:.1} us", avg_get_us);
    println!("avg put:     {:.1} us", avg_put_us);
    println!(
        "get p50/95/99: {:.1} / {:.1} / {:.1} us",
        g50 as f64 / 1000.0,
        g95 as f64 / 1000.0,
        g99 as f64 / 1000.0,
    );
    println!(
        "put p50/95/99: {:.1} / {:.1} / {:.1} us",
        p50 as f64 / 1000.0,
        p95 as f64 / 1000.0,
        p99 as f64 / 1000.0,
    );
    println!();

    db.close().await.expect("close db");
}

fn generate_keys(n: u64, key_len: usize) -> Arc<Vec<Vec<u8>>> {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x1234_5678);
    let mut keys: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let mut k = vec![0u8; key_len];
            rng.fill(&mut k[..]);
            k
        })
        .collect();
    keys.shuffle(&mut rng);
    Arc::new(keys)
}

fn pct(samples: &[u64], q: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let idx = ((samples.len() - 1) as f64 * q) as usize;
    samples[idx]
}

/// Adds a fixed per-request delay to simulate a remote object store. The
/// delay applies to GET/HEAD/PUT/DELETE/LIST equally; we don't model
/// upload/download bandwidth.
#[derive(Debug)]
struct SlowObjectStore {
    inner: Arc<dyn ObjectStore>,
    delay: std::time::Duration,
}

impl std::fmt::Display for SlowObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlowObjectStore({:?}, {})", self.delay, self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for SlowObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        tokio::time::sleep(self.delay).await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        tokio::time::sleep(self.delay).await;
        self.inner.put_multipart(location).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        tokio::time::sleep(self.delay).await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, opts: GetOptions) -> object_store::Result<GetResult> {
        tokio::time::sleep(self.delay).await;
        self.inner.get_opts(location, opts).await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> object_store::Result<Bytes> {
        tokio::time::sleep(self.delay).await;
        self.inner.get_range(location, range).await
    }

    async fn head(&self, location: &Path) -> object_store::Result<ObjectMeta> {
        tokio::time::sleep(self.delay).await;
        self.inner.head(location).await
    }

    async fn delete(&self, location: &Path) -> object_store::Result<()> {
        tokio::time::sleep(self.delay).await;
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        // List streams are not delayed per-item; only the call setup is
        // simulated as a sync no-op. This matches what cache reads care
        // about (initial latency dominates).
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        tokio::time::sleep(self.delay).await;
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        tokio::time::sleep(self.delay).await;
        self.inner.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        tokio::time::sleep(self.delay).await;
        self.inner.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        tokio::time::sleep(self.delay).await;
        self.inner.copy_if_not_exists(from, to).await
    }

    async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        tokio::time::sleep(self.delay).await;
        self.inner.rename_if_not_exists(from, to).await
    }
}
