use crate::cached_object_store::stats::CachedObjectStoreStats;
use crate::cached_object_store::storage_fs::FsCacheStorage;
use crate::cached_object_store::LocalCacheEntry;
use crate::cached_object_store::options::ObjectStoreCacheOptions;
use crate::rand::DbRand;
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, stream, stream::BoxStream, StreamExt};
use object_store::{path::Path, GetOptions, GetResult, ObjectMeta, ObjectStore};
use object_store::{
    Attributes, CopyOptions, GetRange, GetResultPayload, PutMultipartOptions, PutResult,
    RenameOptions,
};
use object_store::{ListResult, MultipartUpload, PutOptions, PutPayload};
use slatedb_common::clock::SystemClock;
use std::{ops::Range, sync::Arc};
use tokio::sync::OnceCell;

use crate::single_flight::SingleFlight;

use crate::cached_object_store::admission::AdmissionPicker;
use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
use crate::error::SlateDBError;
use crate::object_store_intent::{get_read_intent, get_write_intent, ReadKind, WriteKind};
use log::warn;

use slatedb_common::metrics::MetricsRecorderHelper;

#[derive(Debug, Clone)]
pub struct CachedObjectStore {
    object_store: Arc<dyn ObjectStore>,
    pub(crate) part_size_bytes: usize, // expected to be aligned with mb or kb
    pub(crate) cache_storage: Arc<dyn LocalCacheStorage>,
    pub(crate) admission_picker: AdmissionPicker,
    pub(crate) cache_puts: bool,
    // Absolute path of the root folder relative to the bucket. See #1319.
    resolved_root: Arc<OnceCell<Path>>,
    stats: Arc<CachedObjectStoreStats>,
    // Deduplicates concurrent HEAD requests for the same path after a cache miss.
    head_flights: SingleFlight<Path, (ObjectMeta, Attributes)>,
    // Deduplicates concurrent prefetch/GET requests for the same path after a cache miss.
    prefetch_flights: SingleFlight<(Path, Option<GetRangeKey>), (ObjectMeta, Attributes)>,
    // Deduplicates concurrent fetches of the same part after a cache miss.
    // Keyed on (path, part_id) so multiple readers needing the same part share one fetch.
    part_flights: SingleFlight<(Path, PartID), Bytes>,
}

impl CachedObjectStore {
    pub(crate) fn new(
        object_store: Arc<dyn ObjectStore>,
        cache_storage: Arc<dyn LocalCacheStorage>,
        part_size_bytes: usize,
        cache_puts: bool,
        stats: Arc<CachedObjectStoreStats>,
    ) -> Result<Arc<Self>, SlateDBError> {
        if part_size_bytes == 0 || !part_size_bytes.is_multiple_of(1024) {
            return Err(SlateDBError::InvalidCachePartSize);
        }

        Ok(Arc::new(Self {
            object_store,
            part_size_bytes,
            cache_storage,
            stats,
            admission_picker: AdmissionPicker::default(),
            cache_puts,
            resolved_root: Arc::new(OnceCell::new()),
            head_flights: SingleFlight::new(),
            prefetch_flights: SingleFlight::new(),
            part_flights: SingleFlight::new(),
        }))
    }

    pub(crate) async fn start_evictor(&self) {
        self.cache_storage.start_evictor().await;
    }

    /// Start building a `CachedObjectStore` over `object_store`. The
    /// returned [`CachedObjectStoreBuilder`] requires `.root_folder(..)`
    /// before `.build()`; everything else has a sensible default.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = CachedObjectStore::builder(backend)
    ///     .root_folder("/var/cache/slatedb")
    ///     .build()
    ///     .await?;
    /// let db = Db::builder(path, cache).build().await?;
    /// ```
    pub fn builder(object_store: Arc<dyn ObjectStore>) -> CachedObjectStoreBuilder {
        CachedObjectStoreBuilder::new(object_store)
    }

    /// Build a `CachedObjectStore` from `ObjectStoreCacheOptions`, returning `None`
    /// if caching is not configured (i.e. `root_folder` is `None`). When `Some` is
    /// returned the evictor has already been started.
    pub(crate) async fn from_config(
        object_store: Arc<dyn ObjectStore>,
        options: &ObjectStoreCacheOptions,
        recorder: &MetricsRecorderHelper,
        clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
    ) -> Result<Option<Arc<Self>>, SlateDBError> {
        let cache_root_folder = match &options.root_folder {
            None => return Ok(None),
            Some(f) => f,
        };
        let stats = Arc::new(CachedObjectStoreStats::new(recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_root_folder.clone(),
            options.max_cache_size_bytes,
            options.scan_interval,
            stats.clone(),
            clock,
            rand,
            options.max_open_file_handles,
        ));
        let cached = Self::new(
            object_store,
            cache_storage,
            options.part_size_bytes,
            options.cache_puts,
            stats,
        )?;
        cached.start_evictor().await;
        Ok(Some(cached))
    }

    /// Returns the canonical cache key for a requested location.
    ///
    /// The key is `resolved_root + location` once root resolution succeeds.
    /// Returns `None` while the root is still unresolved. The root is resolved
    /// lazily from observed metadata locations, so this method may return `None`
    /// for early requests until successful GET/HEAD responses are observed.
    fn cache_location_for(&self, location: &Path) -> Option<Path> {
        cache_location_for(&self.resolved_root, location)
    }

    /// Drop any cached entry for `location`. Called from the read path
    /// when the caller signals (via `ReadIntent.retry`) that a previous
    /// read returned corrupt bytes — the cached copy is suspect and
    /// must not be served again. No-op when the cache hasn't resolved
    /// its root yet or `location` has no derivable cache key.
    async fn evict_entry(&self, location: &Path) {
        let Some(cache_location) = self.cache_location_for(location) else {
            return;
        };
        let entry = self
            .cache_storage
            .entry(&cache_location, self.part_size_bytes);
        entry.delete().await;
    }

    /// Lazily resolves the root prefix and validates the derived cache key.
    ///
    /// ## Arguments
    ///
    /// - `requested_location`: the location from the incoming request, treated as a suffix
    ///   for root inference.
    /// - `meta_location`: the location from the observed metadata, expected to be
    ///   `root + requested_location`.
    ///
    /// ## Returns
    ///
    /// Returns `true` only when:
    ///
    /// - `resolved_root` is already known or can be safely inferred from metadata; and
    /// - the derived canonical cache key matches `meta_location`.
    ///
    /// Returns `false` otherwise. This is a defensive guard: on mismatch, cache writes are
    /// skipped to avoid poisoning cache entries with unsafe keys.
    fn resolve_root(&self, requested_location: &Path, meta_location: &Path) -> bool {
        // If root is not resolved yet, try to infer it from the metadata location
        if self.resolved_root.get().is_none() {
            let Some(root) = Self::infer_root(requested_location, meta_location) else {
                warn!(
                    "failed to resolve cache root lazily [requested_location={}, meta_location={}]",
                    requested_location, meta_location,
                );
                return false;
            };
            let _ = self.resolved_root.set(root);
        }
        // Get cache location so we can verify it matches the metadata location. This should always
        // succeed after root resolution.
        let Some(cache_location) = self.cache_location_for(requested_location) else {
            warn!(
                "cache location is unexpectedly unavailable after root resolution [requested_location={}, meta_location={}]",
                requested_location, meta_location,
            );
            return false;
        };
        // Verify the cache location matches the metadata location. Again, should always be true, but
        // you can never trust object stores completely.
        if &cache_location != meta_location {
            warn!(
                "resolved root mismatch [requested_location={}, cache_location={}, meta_location={}]",
                requested_location, cache_location, meta_location,
            );
            return false;
        }
        true
    }

    /// Infers a root prefix by treating `requested_location` as a suffix of `meta_location`.
    ///
    /// Returns `Some(root)` when `meta_location == root + requested_location`,
    /// otherwise returns `None`.
    fn infer_root(requested_location: &Path, meta_location: &Path) -> Option<Path> {
        let requested_str = requested_location.as_ref();
        let meta_str = meta_location.as_ref();

        if requested_str.is_empty() {
            return Some(meta_location.clone());
        }

        let prefix = meta_str.strip_suffix(requested_str)?;

        // Ensure suffix matching happens at a path-segment boundary.
        if !prefix.is_empty() && !prefix.ends_with('/') {
            return None;
        }

        Some(Path::from(prefix.trim_end_matches('/')))
    }

    #[cfg(test)]
    pub(crate) async fn cached_head(&self, location: &Path) -> object_store::Result<GetResult> {
        self.cached_head_with_options(location, GetOptions::default().with_head(true))
            .await
    }

    async fn cached_head_with_options(
        &self,
        location: &Path,
        mut opts: GetOptions,
    ) -> object_store::Result<GetResult> {
        opts.range = None;
        opts.head = true;

        if let Some(cache_location) = self.cache_location_for(location) {
            let entry = self
                .cache_storage
                .entry(&cache_location, self.part_size_bytes);
            if let Ok(Some((meta, attributes))) = entry.read_head().await {
                return Ok(head_only_get_result(meta, attributes));
            }
        }

        // Cache miss — deduplicate concurrent HEAD requests for the same path.
        let (meta, attributes) = self
            .head_flights
            .call(location.clone(), || async {
                let result = self.object_store.get_opts(location, opts).await?;
                let meta = result.meta.clone();
                let attributes = result.attributes.clone();
                if self.resolve_root(location, &meta.location) {
                    self.save_get_result(result).await.ok();
                }
                Ok::<_, object_store::Error>((meta, attributes))
            })
            .await?;
        Ok(head_only_get_result(meta, attributes))
    }

    pub(crate) async fn cached_get_opts(
        &self,
        location: &Path,
        opts: GetOptions,
    ) -> object_store::Result<GetResult> {
        let (meta, attributes) = self.maybe_prefetch_range(location, opts.clone()).await?;

        // If we still can't derive a safe canonical cache key after calling
        // maybe_prefetch_range, bypass cache for this request since there's no
        // point in fetching by range.
        if self.cache_location_for(location).is_none() {
            return self.object_store.get_opts(location, opts.clone()).await;
        }

        let get_range = opts.range.clone();
        let range = self.canonicalize_range(get_range, meta.size)?;
        let parts = self.split_range_into_parts(range.clone());
        let extensions = opts.extensions;

        // read parts, and concatenate them into a single stream. please note that
        // some of these part may not be cached, we'll still fallback to the object
        // store to get the missing parts.
        let futures = parts
            .into_iter()
            .map(|(part_id, range_in_part)| {
                self.read_part(location, part_id, range_in_part, extensions.clone())
            })
            .collect::<Vec<_>>();
        let result_stream = stream::iter(futures).then(|fut| fut).boxed();

        Ok(GetResult {
            meta,
            range,
            attributes,
            payload: GetResultPayload::Stream(result_stream),
        })
    }

    async fn cached_put_opts(
        &self,
        location: &Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<PutResult> {
        // Only cache if the cache_puts option is enabled.
        // Also skip caching for write intents that aren't worth admitting:
        //   - Wal: short-lived, fenced and superseded by L0 flush.
        //   - Manifest: small metadata, the cache would churn over rapid
        //     manifest versions for no benefit.
        //   - CompactionOutput: compactor produces large SSTs in bulk,
        //     admitting them would evict hotter L0 entries.
        // Flush (memtable to L0) and untagged puts fall through and are
        // written to the upstream store and the cache, preserving the
        // prior behavior for any caller that doesn't tag.
        let skip_for_intent = matches!(
            get_write_intent(&opts.extensions).map(|i| i.kind),
            Some(WriteKind::Wal | WriteKind::Manifest | WriteKind::CompactionOutput)
        );
        if !self.cache_puts || skip_for_intent {
            // If caching is disabled, just write to the upstream object store without cloning
            return self.object_store.put_opts(location, payload, opts).await;
        }

        // First, write to the upstream object store (cloning payload is cheap since it's just a Arc internally)
        let result = self
            .object_store
            .put_opts(location, payload.clone(), opts)
            .await?;

        // Then, save to local cache if admission policy allows it
        let Some(cache_location) = self.cache_location_for(location) else {
            return Ok(result);
        };
        let entry = self
            .cache_storage
            .entry(&cache_location, self.part_size_bytes);
        if self.admission_picker.pick(entry.as_ref()).admitted() {
            // Convert PutPayload to stream and save parts to cache.
            // Note: cached_head() already saved the head, so we only need to save parts
            let stream = stream::iter(payload.into_iter()).map(Ok::<Bytes, object_store::Error>);

            // Save parts only, ignoring errors (cache failures shouldn't fail the PUT operation)
            self.save_parts_stream(entry, stream, 0).await.ok();
        }

        Ok(result)
    }

    // if an object is not cached before, maybe_prefetch_range will try to prefetch the object from the
    // object store and save the parts into the local disk cache. the prefetching is helpful to reduce the
    // number of GET requests to the object store, it'll try to aggregate the parts among the range into a
    // single GET request, and save the related parts into local disks together.
    // when it sends GET requests to the object store, the range is expected to be ALIGNED with the part
    // size.
    async fn maybe_prefetch_range(
        &self,
        location: &Path,
        mut opts: GetOptions,
    ) -> object_store::Result<(ObjectMeta, Attributes)> {
        if let Some(cache_location) = self.cache_location_for(location) {
            let entry = self
                .cache_storage
                .entry(&cache_location, self.part_size_bytes);
            match entry.read_head().await {
                Ok(Some((meta, attrs))) => return Ok((meta, attrs)),
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        "failed to read head from disk cache, will fallback to object store [location={}, error={:?}]",
                        location, e,
                    );
                }
            }
        }

        if let Some(range) = &opts.range {
            opts.range = Some(self.align_get_range(range));
        }

        // Cache miss — deduplicate concurrent prefetch requests for the same path.
        // Only one caller performs the fetch+save; others share the metadata result.
        // Parts not covered by the winning caller's range are handled by read_part's
        // own object-store fallback, so correctness is maintained.
        self.prefetch_flights
            .call(
                (location.clone(), opts.range.clone().map(Into::into)),
                || async {
                    let get_result = self.object_store.get_opts(location, opts).await?;
                    let result_meta = get_result.meta.clone();
                    let result_attrs = get_result.attributes.clone();
                    // swallow the error on saving to disk here (the disk might be already full), just fallback
                    // to the object store.
                    // TODO: add a warning log here
                    if self.resolve_root(location, &result_meta.location) {
                        self.save_get_result(get_result).await.ok();
                    }
                    Ok((result_meta, result_attrs))
                },
            )
            .await
    }

    /// save the GetResult to the disk cache, a GetResult may be transformed into multiple part
    /// files and a meta file. please note that the `range` in the GetResult is expected to be
    /// aligned with the part size.
    async fn save_get_result(&self, result: GetResult) -> object_store::Result<u64> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        assert!(result.range.start.is_multiple_of(part_size_bytes_u64));
        assert!(
            result.range.end.is_multiple_of(part_size_bytes_u64)
                || result.range.end == result.meta.size
        );

        let entry = self
            .cache_storage
            .entry(&result.meta.location, self.part_size_bytes);
        let object_size = result.meta.size;

        if self.admission_picker.pick(entry.as_ref()).admitted() {
            entry.save_head((&result.meta, &result.attributes)).await?;

            let start_part_number = usize::try_from(result.range.start / part_size_bytes_u64)
                .expect("Part number exceeds u32 on a 32-bit system. Try increasing part size.");

            let stream = result.into_stream();

            self.save_parts_stream(entry, stream, start_part_number)
                .await?;
        }

        Ok(object_size)
    }

    /// Save a stream of bytes to cache as parts, starting from the specified part number.
    /// Returns the number of bytes saved.
    /// This method only saves the data parts - the head should be saved separately.
    async fn save_parts_stream<S>(
        &self,
        entry: Box<dyn LocalCacheEntry>,
        mut stream: S,
        start_part_number: usize,
    ) -> object_store::Result<usize>
    where
        S: stream::Stream<Item = Result<Bytes, object_store::Error>> + Unpin,
    {
        let mut buffer = BytesMut::new();
        let mut part_number = start_part_number;
        let mut total_bytes: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            total_bytes += chunk.len();
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= self.part_size_bytes {
                let to_write = buffer.split_to(self.part_size_bytes);
                entry.save_part(part_number, to_write.into()).await?;
                part_number += 1;
            }
        }

        // Save any remaining bytes as the last part
        if !buffer.is_empty() {
            entry.save_part(part_number, buffer.into()).await?;
        }

        Ok(total_bytes)
    }

    // split the range into parts, and return the part id and the range inside the part.
    fn split_range_into_parts(&self, range: Range<u64>) -> Vec<(PartID, Range<usize>)> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        let range_aligned = self.align_range(&range, self.part_size_bytes);
        let start_part = range_aligned.start / part_size_bytes_u64;
        let end_part = range_aligned.end / part_size_bytes_u64;
        let mut parts: Vec<_> = (start_part..end_part)
            .map(|part_id| {
                (
                    usize::try_from(part_id).expect("Number of parts exceeds usize"),
                    Range {
                        start: 0,
                        end: self.part_size_bytes,
                    },
                )
            })
            .collect();
        if parts.is_empty() {
            return vec![];
        }
        if let Some(first_part) = parts.first_mut() {
            first_part.1.start = usize::try_from(range.start % part_size_bytes_u64)
                .expect("Part size is too large to fit in a usize");
        }
        if let Some(last_part) = parts.last_mut() {
            if !range.end.is_multiple_of(part_size_bytes_u64) {
                last_part.1.end = usize::try_from(range.end % part_size_bytes_u64)
                    .expect("Part size is too large to fit in a usize");
            }
        }
        parts
    }

    /// get from disk if the parts are cached, otherwise start a new GET request.
    /// the io errors on reading the disk caches will be ignored, just fallback to
    /// the object store.
    fn read_part(
        &self,
        location: &Path,
        part_id: PartID,
        range_in_part: Range<usize>,
        extensions: object_store::Extensions,
    ) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let this = self.clone();
        let location = location.clone();
        Box::pin(async move {
            this.stats.object_store_cache_part_access.increment(1);

            // Try local cache first.
            if let Some(cache_location) = this.cache_location_for(&location) {
                let entry = this
                    .cache_storage
                    .entry(&cache_location, this.part_size_bytes);
                // Cache hit, so return immediately.
                if let Ok(Some(bytes)) = entry.read_part(part_id, range_in_part.clone()).await {
                    this.stats.object_store_cache_part_hits.increment(1);
                    return Ok(bytes);
                }
            }

            // Cache miss, so we need to fetch from the object store.
            // Read Part — deduplicate concurrent fetches of the same part.
            // The SingleFlight fetches the full part and saves it to cache; each
            // caller then slices out their own range_in_part.
            let bytes = this
                .part_flights
                .call((location.clone(), part_id), || async {
                    let part_range = Range {
                        start: (part_id * this.part_size_bytes) as u64,
                        end: ((part_id + 1) * this.part_size_bytes) as u64,
                    };
                    let get_result = this
                        .object_store
                        .get_opts(
                            &location,
                            GetOptions {
                                range: Some(GetRange::Bounded(part_range)),
                                extensions: extensions.clone(),
                                ..Default::default()
                            },
                        )
                        .await?;

                    // Get the cache entry again after successful get so we can cache the part.
                    let cache_entry = if this.resolve_root(&location, &get_result.meta.location) {
                        this.cache_location_for(&location).map(|cache_location| {
                            this.cache_storage
                                .entry(&cache_location, this.part_size_bytes)
                        })
                    } else {
                        // If the root resolution fails, we won't be able to derive a canonical cache
                        // key. Skip saving to cache to avoid poisoning the cache with unsafe keys.
                        None
                    };

                    // Save the head and the part to cache for future accesses. Just read the bytes
                    // if we still can't derive a canonical cache key.
                    let bytes = if let Some(entry) = cache_entry {
                        // Save the head and the part to cache for future accesses.
                        let meta = get_result.meta.clone();
                        let attrs = get_result.attributes.clone();
                        let bytes = get_result.bytes().await?;
                        entry.save_head((&meta, &attrs)).await.ok();
                        entry.save_part(part_id, bytes.clone()).await.ok();
                        bytes
                    } else {
                        get_result.bytes().await?
                    };

                    Ok::<_, object_store::Error>(bytes)
                })
                .await?;

            Ok(Bytes::copy_from_slice(&bytes.slice(range_in_part)))
        })
    }

    // given the range and object size, return the canonicalized `Range<usize>` with concrete start and
    // end.
    fn canonicalize_range(
        &self,
        range: Option<GetRange>,
        object_size: u64,
    ) -> object_store::Result<Range<u64>> {
        let (start_offset, end_offset) = match range {
            None => (0, object_size),
            Some(range) => match range {
                GetRange::Bounded(range) => {
                    if range.start >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: range.start,
                                length: object_size,
                            }),
                        });
                    }
                    if range.start >= range.end {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::Inconsistent {
                                start: range.start,
                                end: range.end,
                            }),
                        });
                    }
                    (range.start, range.end.min(object_size))
                }
                GetRange::Offset(offset) => {
                    if offset >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: offset,
                                length: object_size,
                            }),
                        });
                    }
                    (offset, object_size)
                }
                GetRange::Suffix(suffix) => (object_size.saturating_sub(suffix), object_size),
            },
        };
        Ok(Range {
            start: start_offset,
            end: end_offset,
        })
    }

    fn align_get_range(&self, range: &GetRange) -> GetRange {
        match range {
            GetRange::Bounded(bounded) => {
                let aligned = self.align_range(bounded, self.part_size_bytes);
                GetRange::Bounded(aligned)
            }
            GetRange::Suffix(suffix) => {
                let suffix_aligned = self.align_range(&(0..*suffix), self.part_size_bytes).end;
                GetRange::Suffix(suffix_aligned)
            }
            GetRange::Offset(offset) => {
                let offset_aligned = *offset - *offset % self.part_size_bytes as u64;
                GetRange::Offset(offset_aligned)
            }
        }
    }

    fn align_range(&self, range: &Range<u64>, alignment: usize) -> Range<u64> {
        let alignment = alignment as u64;
        let start_aligned = range.start - range.start % alignment;
        let end_aligned = range.end.div_ceil(alignment) * alignment;
        Range {
            start: start_aligned,
            end: end_aligned,
        }
    }
}

/// Builder for [`CachedObjectStore`]. Returned by
/// [`CachedObjectStore::builder`]. `root_folder` is required;
/// everything else has a sensible default. Internal SlateDB types
/// (metrics recorder, system clock, RNG) default to
/// `MetricsRecorderHelper::noop()`, a real wall-clock, and a fresh
/// `DbRand` respectively; callers only need to override them for
/// tests or to share state with the rest of a SlateDB instance.
pub struct CachedObjectStoreBuilder {
    object_store: Arc<dyn ObjectStore>,
    options: ObjectStoreCacheOptions,
    recorder: Option<MetricsRecorderHelper>,
    clock: Option<Arc<dyn SystemClock>>,
    rand: Option<Arc<DbRand>>,
}

impl CachedObjectStoreBuilder {
    fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            object_store,
            options: ObjectStoreCacheOptions::default(),
            recorder: None,
            clock: None,
            rand: None,
        }
    }

    /// Required. Root directory under which part files are written.
    pub fn root_folder(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.options.root_folder = Some(path.into());
        self
    }

    /// Whether to write through PUTs to the local cache. Default `false`
    /// (cache populates on read miss only).
    pub fn cache_puts(mut self, enabled: bool) -> Self {
        self.options.cache_puts = enabled;
        self
    }

    /// Size of each on, disk part file. Must be a multiple of 1 KB.
    /// Default 4 MB.
    pub fn part_size_bytes(mut self, bytes: usize) -> Self {
        self.options.part_size_bytes = bytes;
        self
    }

    /// Hard cap on local cache size, in bytes. Default 16 GB on 64, bit
    /// systems, 4 GB on 32, bit systems.
    pub fn max_cache_size_bytes(mut self, bytes: usize) -> Self {
        self.options.max_cache_size_bytes = Some(bytes);
        self
    }

    /// How often the evictor scans the cache directory to rebuild its
    /// in, memory map. `None` scans once at startup only. Default
    /// `Some(1 hour)`.
    pub fn scan_interval(mut self, interval: Option<std::time::Duration>) -> Self {
        self.options.scan_interval = interval;
        self
    }

    /// Maximum open file handles for cache parts. Default 1000.
    pub fn max_open_file_handles(mut self, n: usize) -> Self {
        self.options.max_open_file_handles = n;
        self
    }

    /// Replace the full options struct in one go. Convenient when the
    /// caller already has an `ObjectStoreCacheOptions` (for example,
    /// parsed from external config). Setter methods called after this
    /// override individual fields.
    pub fn options(mut self, options: ObjectStoreCacheOptions) -> Self {
        self.options = options;
        self
    }

    /// Override the metrics recorder. Defaults to
    /// `MetricsRecorderHelper::noop()`. Override only if you want the
    /// cache's stats to feed a specific recorder (e.g. the same one
    /// passed to `Db::builder`).
    pub fn metrics_recorder(mut self, recorder: MetricsRecorderHelper) -> Self {
        self.recorder = Some(recorder);
        self
    }

    /// Override the system clock. Defaults to `DefaultSystemClock`.
    /// Tests use this to plug in a mock clock.
    pub fn system_clock(mut self, clock: Arc<dyn SystemClock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Override the random number generator. Defaults to
    /// `DbRand::default()`. Tests use this for deterministic seeding.
    pub fn rand(mut self, rand: Arc<DbRand>) -> Self {
        self.rand = Some(rand);
        self
    }

    /// Construct the [`CachedObjectStore`]. Errors if `root_folder` is
    /// not set, or if the part size is invalid.
    pub async fn build(self) -> Result<Arc<CachedObjectStore>, crate::Error> {
        if self.options.root_folder.is_none() {
            return Err(crate::Error::invalid(
                "CachedObjectStore::builder() requires root_folder".to_string(),
            ));
        }
        let recorder = self.recorder.unwrap_or_else(MetricsRecorderHelper::noop);
        let clock = self
            .clock
            .unwrap_or_else(|| Arc::new(slatedb_common::clock::DefaultSystemClock::new()));
        let rand = self.rand.unwrap_or_else(|| Arc::new(DbRand::default()));
        match CachedObjectStore::from_config(
            self.object_store,
            &self.options,
            &recorder,
            clock,
            rand,
        )
        .await
        .map_err(crate::Error::from)?
        {
            Some(cache) => Ok(cache),
            None => Err(crate::Error::invalid(
                "CachedObjectStore::builder() requires root_folder".to_string(),
            )),
        }
    }
}

fn head_only_get_result(meta: ObjectMeta, attributes: Attributes) -> GetResult {
    GetResult {
        payload: GetResultPayload::Stream(stream::empty().boxed()),
        range: 0..0,
        meta,
        attributes,
    }
}

fn cache_location_for(resolved_root: &Arc<OnceCell<Path>>, location: &Path) -> Option<Path> {
    resolved_root.get().map(|root| {
        if root.as_ref().is_empty() {
            return location.clone();
        }
        root.parts().chain(location.parts()).collect()
    })
}

impl std::fmt::Display for CachedObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CachedObjectStore({}, {})",
            self.object_store, self.cache_storage
        )
    }
}

#[async_trait::async_trait]
impl ObjectStore for CachedObjectStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        // Head requests stay cache-first regardless of intent: they're
        // small and frequently-accessed metadata reads where the cost of
        // a HEAD is much higher than the cost of the cached bytes.
        if options.head {
            return self.cached_head_with_options(location, options).await;
        }

        // ReadIntent routing for non-head reads:
        //   - CompactionInput: bypass the cache entirely. Compaction
        //     scans are one-shot reads that would pollute the cache
        //     with bytes that won't be re-read.
        //   - retry = Some(..): the caller saw a recoverable decode
        //     error (CRC mismatch / decompression / block-decode
        //     failure) on a previous read. Drop our cached copy before
        //     refetching from upstream so we don't serve the same
        //     bytes again.
        // Anything else (Foreground, Warmup, untagged) takes the
        // existing cache-aware path unchanged.
        let read_intent = get_read_intent(&options.extensions);
        if matches!(read_intent.map(|i| i.kind), Some(ReadKind::CompactionInput)) {
            return self.object_store.get_opts(location, options).await;
        }
        if read_intent.and_then(|i| i.retry).is_some() {
            self.evict_entry(location).await;
        }
        self.cached_get_opts(location, options).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.cached_put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.object_store.put_multipart_opts(location, opts).await
    }

    /// Deletion of the cache entries associated with the object being
    /// deleted is not atomic with respect to the object deletion from
    /// the underlying object store. So for some period of time after
    /// the deletion, cached object parts are still visible in the cache.
    /// But assuming each object ever created by SlateDB is immutable and
    /// has a unique name, this is not a problem.
    ///
    /// If eviction is enabled, deletion of the associated cache entries
    /// happens asynchronously; when the control returns to the caller,
    /// the entries still might be present in the cache. If eviction is
    /// off, the deletion happens synchronously; when the control returns
    /// to the caller, it is guaranteed no entries present in the cache
    /// (assuming no errors happened during the deletion).
    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        let resolved_root = self.resolved_root.clone();
        let cache_storage = self.cache_storage.clone();
        let part_size_bytes = self.part_size_bytes;

        self.object_store
            .delete_stream(locations)
            .then(move |result| {
                let resolved_root = resolved_root.clone();
                let cache_storage = cache_storage.clone();
                async move {
                    if let Ok(ref location) = result {
                        if let Some(cache_location) = cache_location_for(&resolved_root, location) {
                            let entry = cache_storage.entry(&cache_location, part_size_bytes);
                            entry.delete().await;
                        }
                    }
                    result
                }
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.object_store.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.object_store.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.object_store.rename_opts(from, to, options).await
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidGetRange {
    #[error("Range start too large, requested: {requested}, length: {length}")]
    StartTooLarge { requested: u64, length: u64 },

    #[error("Range started at {start} and ended at {end}")]
    Inconsistent { start: u64, end: u64 },
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
/// A mirror of [`object_store::GetRange`] that implements [`Hash`] and [`Eq`],
/// allowing it to be used as a key in hash-based collections (e.g. `SingleFlight`).
enum GetRangeKey {
    Bounded(Range<u64>),
    Offset(u64),
    Suffix(u64),
}

impl From<GetRange> for GetRangeKey {
    fn from(range: GetRange) -> Self {
        match range {
            GetRange::Bounded(r) => GetRangeKey::Bounded(r),
            GetRange::Offset(o) => GetRangeKey::Offset(o),
            GetRange::Suffix(s) => GetRangeKey::Suffix(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use object_store::{
        path::Path, Attributes, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutPayload,
    };
    use rand::Rng;

    use super::CachedObjectStore;
    use crate::cached_object_store::stats::CachedObjectStoreStats;
    use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
    use crate::cached_object_store::storage_fs::FsCacheEntry;
    use crate::cached_object_store::storage_fs::FsCacheStorage;
    use crate::rand::DbRand;
    use crate::test_utils::{gen_rand_bytes, FlakyObjectStore, GatedObjectStore};
    use slatedb_common::clock::DefaultSystemClock;
    use slatedb_common::metrics::MetricsRecorderHelper;

    fn new_test_cache_folder() -> std::path::PathBuf {
        let mut rng = rand::rng();
        let dir_name: String = (0..10)
            .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
            .collect();
        let path = format!("/tmp/testcache-{}", dir_name);
        let _ = std::fs::remove_dir_all(&path);
        std::path::PathBuf::from(path)
    }

    #[derive(Debug)]
    struct MismatchedMetaStore {
        inner: Arc<dyn ObjectStore>,
        bad_location: Path,
    }

    impl MismatchedMetaStore {
        fn new(inner: Arc<dyn ObjectStore>, bad_location: Path) -> Self {
            Self {
                inner,
                bad_location,
            }
        }
    }

    impl std::fmt::Display for MismatchedMetaStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MismatchedMetaStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for MismatchedMetaStore {
        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let mut result = self.inner.get_opts(location, options).await?;
            result.meta.location = self.bad_location.clone();
            Ok(result)
        }

        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[test]
    fn test_infer_root() {
        assert_eq!(
            CachedObjectStore::infer_root(
                &Path::from("manifest/0001.manifest"),
                &Path::from("tenant-a/manifest/0001.manifest")
            ),
            Some(Path::from("tenant-a"))
        );
        assert_eq!(
            CachedObjectStore::infer_root(
                &Path::from("manifest/0001.manifest"),
                &Path::from("other/path")
            ),
            None
        );
        assert_eq!(
            CachedObjectStore::infer_root(&Path::from("b/c"), &Path::from("ab/c")),
            None
        );
    }

    #[tokio::test]
    async fn test_lazy_resolve_root_from_meta_location() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let prefixed: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("tenant-a"),
        ));
        let cached_store =
            CachedObjectStore::new(prefixed, cache_storage, 1024, false, stats).unwrap();

        let relative_location = Path::from("manifest/0001.manifest");
        let full_location = Path::from("tenant-a/manifest/0001.manifest");
        let payload = Bytes::from_static(b"tenant-a-manifest");
        backing_store
            .put(&full_location, PutPayload::from_bytes(payload.clone()))
            .await?;

        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let got = cached_store
            .cached_get_opts(&relative_location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got, payload);
        assert_eq!(
            cached_store.resolved_root.get().cloned(),
            Some(Path::from("tenant-a"))
        );

        let scoped_entry = cached_store.cache_storage.entry(&full_location, 1024);
        assert_eq!(scoped_entry.cached_parts().await?.len(), 1);
        let unscoped_entry = cached_store.cache_storage.entry(&relative_location, 1024);
        assert_eq!(unscoped_entry.cached_parts().await?.len(), 0);

        backing_store.delete(&full_location).await?;
        let got_cached = cached_store
            .cached_get_opts(&relative_location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_cached, payload);
        Ok(())
    }

    #[tokio::test]
    async fn test_shared_cache_with_prefix_stores_does_not_collide() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            {
                let recorder = MetricsRecorderHelper::noop();
                Arc::new(CachedObjectStoreStats::new(&recorder))
            },
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let store_a: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("db-a"),
        ));
        let store_b: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("db-b"),
        ));

        let cached_a = CachedObjectStore::new(store_a, cache_storage.clone(), 1024, false, {
            let recorder = MetricsRecorderHelper::noop();
            Arc::new(CachedObjectStoreStats::new(&recorder))
        })
        .unwrap();
        let cached_b = CachedObjectStore::new(store_b, cache_storage.clone(), 1024, false, {
            let recorder = MetricsRecorderHelper::noop();
            Arc::new(CachedObjectStoreStats::new(&recorder))
        })
        .unwrap();

        let relative = Path::from("manifest/0001.manifest");
        let full_a = Path::from("db-a/manifest/0001.manifest");
        let full_b = Path::from("db-b/manifest/0001.manifest");
        let payload_a = Bytes::from_static(b"tenant-a-data");
        let payload_b = Bytes::from_static(b"tenant-b-data");

        backing_store
            .put(&full_a, PutPayload::from_bytes(payload_a.clone()))
            .await?;
        backing_store
            .put(&full_b, PutPayload::from_bytes(payload_b.clone()))
            .await?;

        let got_a = cached_a
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let got_b = cached_b
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_a, payload_a);
        assert_eq!(got_b, payload_b);
        assert_eq!(
            cached_a.resolved_root.get().cloned(),
            Some(Path::from("db-a"))
        );
        assert_eq!(
            cached_b.resolved_root.get().cloned(),
            Some(Path::from("db-b"))
        );

        let unscoped_entry = cache_storage.entry(&relative, 1024);
        assert_eq!(unscoped_entry.cached_parts().await?.len(), 0);
        let scoped_a = cache_storage.entry(&full_a, 1024);
        let scoped_b = cache_storage.entry(&full_b, 1024);
        assert_eq!(scoped_a.cached_parts().await?.len(), 1);
        assert_eq!(scoped_b.cached_parts().await?.len(), 1);

        backing_store.delete(&full_a).await?;
        backing_store.delete(&full_b).await?;
        let got_cached_a = cached_a
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let got_cached_b = cached_b
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_cached_a, payload_a);
        assert_eq!(got_cached_b, payload_b);
        Ok(())
    }

    #[tokio::test]
    async fn test_meta_location_mismatch_bypasses_cache() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let bad_meta_store: Arc<dyn ObjectStore> = Arc::new(MismatchedMetaStore::new(
            backing_store.clone(),
            Path::from("wrong/root"),
        ));
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(bad_meta_store, cache_storage, 1024, false, stats).unwrap();

        let location = Path::from("data/file.sst");
        let payload = Bytes::from_static(b"payload");
        backing_store
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await?;

        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let got = cached_store
            .cached_get_opts(&location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got, payload);
        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let entry = cached_store.cache_storage.entry(&location, 1024);
        assert_eq!(entry.cached_parts().await?.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_save_result_not_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3 + 32);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let part_size = 1024;
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, part_size, false, stats)
                .unwrap();
        let entry = cached_store.cache_storage.entry(&location, 1024);

        let object_size_hint = cached_store.save_get_result(get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3 + 32);

        // assert the cached meta
        let head = entry.read_head().await?;
        assert_eq!(head.unwrap().0.size, 1024 * 3 + 32);

        // assert the parts
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 4);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );
        // check that the unaligned part was also cached
        assert_eq!(
            entry.read_part(3, 0..32).await?,
            Some(payload.slice(3072..3104))
        );

        // delete part 2, known_cache_size is still known
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1, 3]);

        // delete part 3, known_cache_size become None
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 3, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(3, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_save_result_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;
        let part_size = 1024;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, part_size, false, stats).unwrap();
        let entry = cached_store.cache_storage.entry(&location, part_size);
        let object_size_hint = cached_store.save_get_result(get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 3);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );

        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, part_size);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);

        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 2);
        Ok(())
    }

    #[test]
    fn test_split_range_into_parts() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        struct Test {
            input: (Option<GetRange>, usize),
            expect: Vec<(PartID, std::ops::Range<usize>)>,
        }
        let tests = [
            Test {
                input: (None, 1024 * 3),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024)],
            },
            Test {
                input: (None, 1024 * 3 + 12),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (None, 12),
                expect: vec![(0, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(0..1024)), 1024),
                expect: vec![(0, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024)), 20000),
                expect: vec![(0, 128..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 * 2 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..1024), (2, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2..1024 * 3 + 12)), 200000),
                expect: vec![(2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2 - 2..1024 * 3 + 12)), 20000),
                expect: vec![(1, 1022..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Suffix(128)), 1024),
                expect: vec![(0, 896..1024)],
            },
            Test {
                input: (Some(GetRange::Suffix(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(1, 1016..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(8)), 1024 * 4),
                expect: vec![(0, 8..1024), (1, 0..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(2, 8..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4 + 2),
                expect: vec![(2, 8..1024), (3, 0..1024), (4, 0..2)],
            },
        ];

        for t in tests.iter() {
            let range = cached_store
                .canonicalize_range(t.input.0.clone(), t.input.1 as u64)
                .unwrap();
            let parts = cached_store.split_range_into_parts(range);
            assert_eq!(parts, t.expect, "input: {:?}", t.input);
        }
    }

    #[test]
    fn test_align_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_range(&(9..1025), 1024);
        assert_eq!(aligned, 0..2048);
        let aligned = cached_store.align_range(&(1024 + 1..2048 + 4), 1024);
        assert_eq!(aligned, 1024..3072);
    }

    #[test]
    fn test_align_get_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..1025));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..2048));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(12));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(1024));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(1024));
        assert_eq!(aligned, GetRange::Offset(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(12));
        assert_eq!(aligned, GetRange::Offset(0));
    }

    #[tokio::test]
    async fn test_cached_object_store_impl_object_store() -> object_store::Result<()> {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        let test_path = Path::from("/data/testdata1");
        let test_payload = gen_rand_bytes(1024 * 3 + 2);
        object_store
            .put(&test_path, PutPayload::from_bytes(test_payload.clone()))
            .await?;

        // test get entire object
        let test_ranges = vec![
            Some(GetRange::Offset(260817)),
            None,
            Some(GetRange::Bounded(1000..2048)),
            Some(GetRange::Bounded(1000..260817)),
            Some(GetRange::Suffix(10)),
            Some(GetRange::Suffix(260817)),
            Some(GetRange::Offset(1000)),
            Some(GetRange::Offset(0)),
            Some(GetRange::Offset(1028)),
            Some(GetRange::Offset(260817)),
            Some(GetRange::Offset(1024 * 3 + 2)),
            Some(GetRange::Offset(1024 * 3 + 1)),
            #[allow(clippy::reversed_empty_ranges)]
            Some(GetRange::Bounded(2900..2048)),
            Some(GetRange::Bounded(10..10)),
        ];

        // test get a range
        for range in test_ranges.iter() {
            let want = object_store
                .get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                )
                .await;
            let got = cached_store
                .cached_get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                )
                .await;
            match (want, got) {
                (Ok(want), Ok(got)) => {
                    assert_eq!(want.range, got.range);
                    assert_eq!(want.meta, got.meta);
                    assert_eq!(want.bytes().await?, got.bytes().await?);
                }
                (Err(want), Err(got)) => {
                    if want.to_string().to_lowercase().contains("range") {
                        assert!(got.to_string().to_lowercase().contains("range"));
                    }
                }
                (origin_result, cached_result) => {
                    panic!("expect: {:?}, got: {:?}", origin_result, cached_result);
                }
            }
        }
        Ok(())
    }

    /// Helper to build a CachedObjectStore backed by an InstrumentedObjectStore so
    /// we can assert on the number of actual object-store requests made.
    fn build_instrumented_cached_store(
        inner: Arc<dyn ObjectStore>,
    ) -> (
        Arc<slatedb_common::metrics::DefaultMetricsRecorder>,
        Arc<CachedObjectStore>,
    ) {
        use crate::instrumented_object_store::{InstrumentedObjectStore, ObjectStoreComponent};
        use crate::object_stores::ObjectStoreType;
        use slatedb_common::metrics::test_recorder_helper;

        let (recorder, helper) = test_recorder_helper();
        let instrumented = Arc::new(InstrumentedObjectStore::new(
            inner,
            &helper,
            ObjectStoreComponent::Db,
            ObjectStoreType::Main,
        ));

        let test_cache_folder = new_test_cache_folder();
        let noop_helper = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&noop_helper));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store = CachedObjectStore::new(
            instrumented as Arc<dyn ObjectStore>,
            cache_storage,
            1024,
            false,
            stats,
        )
        .unwrap();

        (recorder, cached_store)
    }

    fn get_request_count(
        recorder: &slatedb_common::metrics::DefaultMetricsRecorder,
        api: &str,
    ) -> i64 {
        use crate::instrumented_object_store::stats::REQUEST_COUNT;
        use slatedb_common::metrics::lookup_metric_with_labels;

        let labels = [
            ("component", "db"),
            ("store_type", "main"),
            ("op", "get"),
            ("api", api),
        ];
        lookup_metric_with_labels(recorder, REQUEST_COUNT, &labels).unwrap_or(0)
    }

    #[tokio::test]
    async fn test_single_flight_deduplicates_concurrent_head_requests() {
        // Set up an object in the backing store.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_head_dedup");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        // Wrap with a gate-controlled store so we can block callers deterministically.
        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch many concurrent head requests for the same path.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move { store.cached_head(&p).await }));
        }

        // Wait until exactly 1 caller arrives at the gate (SingleFlight dedup
        // ensures only one caller reaches the head gate).
        gated.head_gate.wait_for_arrivals(1).await;
        assert_eq!(
            gated.head_gate.arrivals(),
            1,
            "SingleFlight should let only 1 through"
        );

        // Release the gate — success path.
        gated.head_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // SingleFlight should collapse them into a single actual HEAD request.
        let count = get_request_count(&recorder, "head");
        assert_eq!(
            count, 1,
            "expected 1 actual object store request, got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_deduplicates_concurrent_get_opts_requests() {
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_get_dedup");
        let payload = gen_rand_bytes(2048);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.get_opts_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch many concurrent get_opts requests for the same path and range.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let opts = GetOptions {
                    range: Some(GetRange::Bounded(0..1024)),
                    ..Default::default()
                };
                let result = store.cached_get_opts(&p, opts).await?;
                result.bytes().await
            }));
        }

        // Wait for the single winning caller to arrive at the gate.
        gated.get_opts_gate.wait_for_arrivals(1).await;
        assert_eq!(
            gated.get_opts_gate.arrivals(),
            1,
            "SingleFlight should let only 1 through"
        );

        // Release — success.
        gated.get_opts_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // The prefetch SingleFlight should collapse all prefetch GETs into one.
        // Part reads may also be deduplicated. Total GET count should be much less than 10.
        let count = get_request_count(&recorder, "get_range");
        assert!(
            count <= 2,
            "expected at most 2 actual object store requests (prefetch + maybe 1 part), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_allows_independent_paths() {
        // Requests to different paths should NOT be deduplicated.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let paths: Vec<Path> = (0..5)
            .map(|i| Path::from(format!("data/independent_{}", i)))
            .collect();
        for p in &paths {
            mem.put(p, PutPayload::from_bytes(gen_rand_bytes(512)))
                .await
                .unwrap();
        }

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        let mut handles = Vec::new();
        for p in &paths {
            let store = cached_store.clone();
            let p = p.clone();
            handles.push(tokio::spawn(async move { store.cached_head(&p).await }));
        }

        // Each distinct path has its own SingleFlight key, so all 5 should arrive.
        gated.head_gate.wait_for_arrivals(5).await;
        assert_eq!(
            gated.head_gate.arrivals(),
            5,
            "different keys should each pass through SingleFlight independently"
        );

        // Release all.
        gated.head_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Each distinct path should result in its own request.
        let count = get_request_count(&recorder, "head");
        assert_eq!(
            count, 5,
            "expected 5 actual object store requests (one per path), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_different_ranges_are_independent() {
        // Requests with different ranges should be treated as separate flights.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_range_independent");
        let payload = gen_rand_bytes(4096);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.get_opts_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        let ranges = vec![
            Some(GetRange::Bounded(0..1024)),
            Some(GetRange::Bounded(1024..2048)),
            Some(GetRange::Suffix(512)),
        ];

        let mut handles = Vec::new();
        for range in ranges {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let opts = GetOptions {
                    range,
                    ..Default::default()
                };
                store.cached_get_opts(&p, opts).await
            }));
        }

        // Each distinct range maps to a different key, so all 3 should arrive.
        gated.get_opts_gate.wait_for_arrivals(3).await;
        assert_eq!(
            gated.get_opts_gate.arrivals(),
            3,
            "different ranges should each pass through SingleFlight independently"
        );

        // Release all.
        gated.get_opts_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Each distinct range key should trigger its own prefetch request.
        let count = get_request_count(&recorder, "get_range");
        assert!(
            count >= 3,
            "expected at least 3 object store requests (one per distinct range), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_concurrent_callers_see_gate_failure() {
        // When the gate is configured to fail, all concurrent waiters on the
        // same SingleFlight key should receive an error (not hang forever).
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_gate_failure");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (_, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch concurrent head requests.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move { store.cached_head(&p).await }));
        }

        // Wait for the single winning caller to arrive at the gate.
        gated.head_gate.wait_for_arrivals(1).await;

        // Inject failure, then release.
        gated.head_gate.set_error(|| object_store::Error::Generic {
            store: "test",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "injected test failure",
            )),
        });
        gated.head_gate.release();

        // All callers should see an error.
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_err(), "expected error when gate injects failure");
        }
    }

    #[tokio::test]
    async fn test_single_flight_retries_after_gate_failure() {
        // After a failure, the SingleFlight should not cache the error,
        // allowing the next call to succeed fresh.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_retry_after_fail");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (_, cached_store) = build_instrumented_cached_store(gated.clone());

        // First call — configure failure.
        let store = cached_store.clone();
        let p = path.clone();
        let handle = tokio::spawn(async move { store.cached_head(&p).await });

        gated.head_gate.wait_for_arrivals(1).await;
        gated.head_gate.set_error(|| object_store::Error::Generic {
            store: "test",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "injected test failure",
            )),
        });
        gated.head_gate.release();

        let result = handle.await.unwrap();
        assert!(result.is_err(), "first call should fail");

        // Second call — configure success.
        gated.head_gate.clear_error();
        let store = cached_store.clone();
        let p = path.clone();
        let handle = tokio::spawn(async move { store.cached_head(&p).await });

        gated.head_gate.wait_for_arrivals(2).await;
        gated.head_gate.release();

        let result = handle.await.unwrap();
        assert!(result.is_ok(), "second call should succeed after retry");
    }

    #[tokio::test]
    async fn test_single_flight_part_fetch_with_get_range_failures() {
        // Validates that when fetching parts fails transiently, the SingleFlight
        // does not permanently cache the failure, and retries succeed.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_part_flaky");
        let payload = gen_rand_bytes(4096);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        // Use a FlakyObjectStore that fails the first get_range call.
        let flaky = Arc::new(FlakyObjectStore::new(mem, 0).with_get_range_failures(1));
        let (_, cached_store) = build_instrumented_cached_store(flaky.clone());

        // First, prime metadata via a full get (get_opts doesn't use get_range).
        let prime_opts = GetOptions {
            range: None,
            ..Default::default()
        };
        let result = cached_store
            .cached_get_opts(&path, prime_opts)
            .await
            .unwrap();
        let _ = result.bytes().await.unwrap();

        // Now try reading — the parts should be cached from the full get above,
        // so even though get_range is flaky, we should succeed from cache.
        let opts = GetOptions {
            range: Some(GetRange::Bounded(0..512)),
            ..Default::default()
        };
        let result = cached_store.cached_get_opts(&path, opts).await;
        assert!(result.is_ok());
        let bytes = result.unwrap().bytes().await.unwrap();
        assert_eq!(&bytes[..], &payload[..512]);
    }

    #[rstest::rstest]
    #[case::no_evictor_cached(false, true)]
    #[case::with_evictor_cached(true, true)]
    #[case::no_evictor_uncached(false, false)]
    #[case::with_evictor_uncached(true, false)]
    #[tokio::test]
    async fn test_delete(#[case] evictor: bool, #[case] cached: bool) {
        const PART_SIZE: usize = 1024;

        let location1 = Path::from("/data/testfile1");
        let location2 = Path::from("/data/testfile2");

        let test_cache_folder = new_test_cache_folder();
        let payload = gen_rand_bytes(PART_SIZE * 3);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));

        object_store
            .put(&location1, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();
        object_store
            .put(&location2, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            evictor.then_some(1024 * 1024),
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store = CachedObjectStore::new(
            object_store,
            Arc::clone(&cache_storage) as Arc<dyn LocalCacheStorage>,
            PART_SIZE,
            false,
            stats,
        )
        .unwrap();
        cached_store.start_evictor().await;

        if cached {
            cached_store
                .get(&location1)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
        }
        cached_store
            .get(&location2)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let cache_location1 = cached_store.cache_location_for(&location1).unwrap();
        let entry1 = cached_store
            .cache_storage
            .entry(&cache_location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        if cached {
            assert_eq!(parts1.len(), 3, "{parts1:?}");
            assert_eq!(cache_storage.file_handle_cache_population(), 6);
        } else {
            assert_eq!(parts1.len(), 0, "{parts1:?}");
            assert_eq!(cache_storage.file_handle_cache_population(), 3);
        }

        let cache_location2 = cached_store.cache_location_for(&location2).unwrap();
        let entry2 = cached_store
            .cache_storage
            .entry(&cache_location2, PART_SIZE);
        let parts2 = entry2.cached_parts().await.unwrap();
        assert_eq!(parts2.len(), 3, "{parts2:?}");

        cached_store.delete(&location1).await.unwrap();
        if evictor {
            // XXX: If evictor is running, deletion is performed asynchronously
            //      from the evictor "thread".
            tokio::time::sleep(Duration::from_secs(3)).await;
        }

        let entry1 = cached_store
            .cache_storage
            .entry(&cache_location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        assert_eq!(parts1.len(), 0, "{parts1:?}");
        assert_eq!(cache_storage.file_handle_cache_population(), 3);

        let cache_location2 = cached_store.cache_location_for(&location2).unwrap();
        let entry2 = cached_store
            .cache_storage
            .entry(&cache_location2, PART_SIZE);
        let parts2 = entry2.cached_parts().await.unwrap();
        assert_eq!(parts2.len(), 3, "{parts2:?}");

        // verify repeated delete is idempotent
        cached_store.delete(&location1).await.unwrap();
        let entry1 = cached_store
            .cache_storage
            .entry(&cache_location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        assert_eq!(parts1.len(), 0, "{parts1:?}");
        assert_eq!(cache_storage.file_handle_cache_population(), 3);
    }

    // ============================================================
    // Intent protocol integration (RFC 0027)
    // ============================================================

    use crate::object_store_intent::testing::IntentRecorderObjectStore;
    use crate::object_store_intent::{
        put_options_with_intent, set_read_intent, ReadIntent, RetryReason, WriteIntent,
    };

    /// Build a `CachedObjectStore` with `cache_puts = true` for the
    /// intent-protocol tests. `part_size = 1024` so a few-KiB payload
    /// gets split across multiple parts without overflowing the in-memory
    /// backend.
    fn build_cached_store_with_puts(inner: Arc<dyn ObjectStore>) -> Arc<CachedObjectStore> {
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        CachedObjectStore::new(inner, cache_storage, 1024, true, stats).unwrap()
    }

    fn get_options_with_read_intent(intent: ReadIntent) -> GetOptions {
        let mut opts = GetOptions::default();
        set_read_intent(&mut opts.extensions, intent);
        opts
    }

    /// `cache_location_for` returns `None` until a read or write through
    /// the cache-aware path resolves the root prefix. The intent-protocol
    /// tests need to inspect the cache directly, so prime the root with
    /// an unrelated Foreground GET first.
    async fn prime_cache_root(cached: &CachedObjectStore, inner: &Arc<dyn ObjectStore>) {
        let warmup_path = Path::from("data/warmup_sentinel");
        inner
            .put(&warmup_path, PutPayload::from_bytes(gen_rand_bytes(256)))
            .await
            .unwrap();
        cached
            .get_opts(
                &warmup_path,
                get_options_with_read_intent(ReadIntent::foreground()),
            )
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }

    /// `WriteIntent::wal()` short-circuits the cache write-through even
    /// when `cache_puts: true`. The upstream write still succeeds.
    #[tokio::test]
    async fn put_with_wal_intent_skips_cache() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cached = build_cached_store_with_puts(inner.clone());
        prime_cache_root(&cached, &inner).await;

        let path = Path::from("data/wal_skipped.sst");
        let payload = gen_rand_bytes(2048);
        cached
            .put_opts(
                &path,
                PutPayload::from_bytes(payload.clone()),
                put_options_with_intent(WriteIntent::wal()),
            )
            .await
            .unwrap();

        // Upstream got the payload.
        let upstream = inner.get(&path).await.unwrap().bytes().await.unwrap();
        assert_eq!(upstream, payload);

        // Cache stayed empty for this path.
        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        let parts = entry.cached_parts().await.unwrap();
        assert!(parts.is_empty(), "expected no parts cached for Wal put");
    }

    /// `WriteIntent::manifest()` skips the cache write-through too.
    #[tokio::test]
    async fn put_with_manifest_intent_skips_cache() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cached = build_cached_store_with_puts(inner.clone());
        prime_cache_root(&cached, &inner).await;

        let path = Path::from("manifest/00000000000000000001.manifest");
        let payload = gen_rand_bytes(2048);
        cached
            .put_opts(
                &path,
                PutPayload::from_bytes(payload.clone()),
                put_options_with_intent(WriteIntent::manifest()),
            )
            .await
            .unwrap();

        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        let parts = entry.cached_parts().await.unwrap();
        assert!(
            parts.is_empty(),
            "expected no parts cached for Manifest put"
        );
    }

    /// `WriteIntent::flush()` keeps the legacy cache-on-write behavior.
    #[tokio::test]
    async fn put_with_flush_intent_caches() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cached = build_cached_store_with_puts(inner.clone());
        prime_cache_root(&cached, &inner).await;

        let path = Path::from("data/l0_cached.sst");
        let payload = gen_rand_bytes(2048);
        cached
            .put_opts(
                &path,
                PutPayload::from_bytes(payload.clone()),
                put_options_with_intent(WriteIntent::flush()),
            )
            .await
            .unwrap();

        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        let parts = entry.cached_parts().await.unwrap();
        assert!(
            !parts.is_empty(),
            "expected Flush put to be cached after write-through"
        );
    }

    /// `ReadIntent::compaction_input()` bypasses the cache entirely:
    /// the response comes from upstream and no parts are admitted, even
    /// after several reads.
    #[tokio::test]
    async fn read_with_compaction_input_intent_bypasses_cache() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cached = build_cached_store_with_puts(inner.clone());
        prime_cache_root(&cached, &inner).await;

        let path = Path::from("data/compaction_input.sst");
        let payload = gen_rand_bytes(2048);
        inner
            .put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        for _ in 0..3 {
            let result = cached
                .get_opts(
                    &path,
                    get_options_with_read_intent(ReadIntent::compaction_input()),
                )
                .await
                .unwrap();
            assert_eq!(result.bytes().await.unwrap(), payload);
        }

        // No admission happened.
        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        let parts = entry.cached_parts().await.unwrap();
        assert!(
            parts.is_empty(),
            "expected no parts cached for CompactionInput reads, found {parts:?}"
        );
    }

    /// When `CachedObjectStore` is stacked above another intent-aware
    /// wrapper, its internal upstream HEAD miss should preserve the
    /// original read intent.
    #[tokio::test]
    async fn head_miss_preserves_read_intent_to_inner_store() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let (recorder, log) = IntentRecorderObjectStore::new(inner.clone());
        let cached = build_cached_store_with_puts(recorder as Arc<dyn ObjectStore>);

        let path = Path::from("data/head_miss.sst");
        inner
            .put(&path, PutPayload::from_bytes(gen_rand_bytes(2048)))
            .await
            .unwrap();

        let mut opts = get_options_with_read_intent(ReadIntent::warmup());
        opts.head = true;
        cached.get_opts(&path, opts).await.unwrap();

        let calls = log.lock().unwrap();
        let head_call = calls
            .iter()
            .find(|c| c.method == "get_opts[head]")
            .expect("expected inner head miss call");
        assert_eq!(head_call.read_intent, Some(ReadIntent::warmup()));
        assert_eq!(head_call.write_intent, None);
    }

    /// If metadata is cached but a data part is missing, the internal
    /// ranged part fetch should also preserve the original read intent
    /// for any wrapper stacked under `CachedObjectStore`.
    #[tokio::test]
    async fn part_miss_preserves_read_intent_to_inner_store() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let (recorder, log) = IntentRecorderObjectStore::new(inner.clone());
        let cached = build_cached_store_with_puts(recorder as Arc<dyn ObjectStore>);

        let path = Path::from("data/part_miss.sst");
        let payload = gen_rand_bytes(2048);
        inner
            .put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        // Resolve the cache root, then seed only the target head. We
        // deliberately do not read the target through `cached_head`,
        // because some object_store backends return bytes for
        // head-shaped `get_opts` calls.
        prime_cache_root(&cached, &inner).await;
        let meta = inner.head(&path).await.unwrap();
        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        entry
            .save_head((&meta, &Attributes::default()))
            .await
            .unwrap();
        log.lock().unwrap().clear();

        let mut range_opts = get_options_with_read_intent(ReadIntent::warmup());
        range_opts.range = Some(GetRange::Bounded(0..512));
        let result = cached.get_opts(&path, range_opts).await.unwrap();
        assert_eq!(result.bytes().await.unwrap(), payload.slice(0..512));

        let calls = log.lock().unwrap();
        let range_call = calls
            .iter()
            .find(|c| c.method.starts_with("get_opts"))
            .unwrap_or_else(|| panic!("expected inner ranged part miss call, saw {calls:?}"));
        assert_eq!(range_call.read_intent, Some(ReadIntent::warmup()));
        assert_eq!(range_call.write_intent, None);
    }

    /// A retry-tagged read evicts the cached entry first, then takes
    /// the normal cached path which refetches from upstream and admits
    /// the fresh bytes. We verify by warming the cache with the OLD
    /// upstream contents, mutating upstream, and confirming a retry-
    /// tagged read returns the NEW contents (cache eviction worked).
    #[tokio::test]
    async fn retry_tagged_read_evicts_cached_entry() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cached = build_cached_store_with_puts(inner.clone());

        let path = Path::from("data/decoded_then_corrupted.sst");
        let original = gen_rand_bytes(2048);
        inner
            .put(&path, PutPayload::from_bytes(original.clone()))
            .await
            .unwrap();

        // Warm the cache with the original payload via a normal Foreground read.
        let warm = cached
            .get_opts(
                &path,
                get_options_with_read_intent(ReadIntent::foreground()),
            )
            .await
            .unwrap();
        assert_eq!(warm.bytes().await.unwrap(), original);

        // Sanity check: cache now has parts for this path.
        let cache_location = cached.cache_location_for(&path).unwrap();
        let entry = cached
            .cache_storage
            .entry(&cache_location, cached.part_size_bytes);
        let pre_parts = entry.cached_parts().await.unwrap();
        assert!(
            !pre_parts.is_empty(),
            "expected cache populated after warm read"
        );

        // Replace the upstream object behind the cache's back. Without
        // eviction, a follow-up read would still serve the stale parts.
        let replacement = gen_rand_bytes(2048);
        inner
            .put(&path, PutPayload::from_bytes(replacement.clone()))
            .await
            .unwrap();

        // Retry-tagged read: must evict + refetch + return new bytes.
        let retried = cached
            .get_opts(
                &path,
                get_options_with_read_intent(
                    ReadIntent::foreground().with_retry(RetryReason::CrcMismatch),
                ),
            )
            .await
            .unwrap();
        assert_eq!(
            retried.bytes().await.unwrap(),
            replacement,
            "retry-tagged read should have refetched fresh bytes from upstream"
        );
    }
}
