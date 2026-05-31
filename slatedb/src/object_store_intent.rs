//! Intent protocol for object store calls (RFC 0026, PR #1653).
//!
//! Two pieces:
//!
//! 1. Public intent types (`WriteIntent`, `ReadIntent`, ...) that
//!    SlateDB attaches to every `ObjectStore` request via the
//!    `Extensions` field on `GetOptions`, `PutOptions`, and
//!    `PutMultipartOptions`.
//! 2. A `testing` submodule with `IntentRecorderObjectStore`, gated by
//!    `cfg(any(test, feature = "test-util"))`. The `tests` submodule
//!    uses it for the verification matrix; external integrators can
//!    use it from their own tests by enabling the `test-util` feature.
//!
//! Per object_store 0.13.2 the known structural gaps are:
//!
//! - `object_store::buffered::BufWriter`: when the buffered payload
//!   fits in capacity, `poll_shutdown` builds `PutOptions` with
//!   `..Default::default()` and drops extensions. Only the multipart
//!   overflow path forwards them. Tracked upstream as
//!   <https://github.com/apache/arrow-rs-object-store/issues/735>. The
//!   two `bufwriter_*_payload_*` matrix tests below document the bug
//!   and act as regression catches for when the upstream fix lands.
//!   In the meantime, SlateDB SST writes below the default 10 MiB
//!   BufWriter capacity reach a wrapper untagged. The focused
//!   `tablestore_with_small_bufwriter_capacity_tags_writes_via_multipart`
//!   test verifies that SlateDB's `new_intent_tagged_bufwriter` helper
//!   attaches extensions correctly by constructing a `TableStore` with
//!   a tiny capacity to force the multipart path.
//! - `head`, `delete`, `list`: no `*_opts` variant exists, so there is
//!   no `Extensions` slot to set. SlateDB rewrites `head(p)` as
//!   `get_opts(p, head_options_with_intent(..))` everywhere intent
//!   matters. `delete` and `list` cannot carry intent at all in 0.13:
//!   the cache wrapper still observes the call (paths flow through),
//!   so eviction-on-delete works, but there is no semantic tag.
//!
//! Coverage today:
//!
//! - WAL writes are tagged `Wal` (do not go through `BufWriter`).
//! - L0 / memtable-flush writes are tagged `Flush` (extension survives
//!   only if the SST is large enough to trigger BufWriter's multipart
//!   overflow; see upstream #735).
//! - Compaction outputs are tagged `CompactionOutput` (same caveat).
//! - Manifest writes (versioned manifest files + the boundary watermark)
//!   are tagged `Manifest`. Same for compactions metadata. Both flow
//!   through `slatedb-txn-obj`, which carries a default `Extensions`
//!   set at construction by [`crate::manifest::store::ManifestStore`]
//!   and [`crate::compactions_store::CompactionsStore`].
//! - Foreground user reads (point, scan, recency, get) are tagged
//!   `Foreground`.
//! - Compaction-input reads issued by the compactor are tagged
//!   `CompactionInput` via `SstIteratorOptions.read_intent`.
//! - Warmup reads issued by `DbCacheManagerOps::warm_sst` are tagged
//!   `Warmup`.
//! - Reissues triggered by `TableStore`'s recoverable validation
//!   errors (CRC mismatch, block decode, decompression failure) carry
//!   `retry = Some(reason)` so a wrapper can drop its cached copy.

use object_store::{Extensions, GetOptions, PutMultipartOptions, PutOptions};

/// What kind of write is being issued.
///
/// Wrapper authors use this to apply admission policy per write kind, e.g.
/// keep data-bearing writes (`Flush`, `CompactionOutput`), skip short-lived
/// or tiny ones (`Wal`, `Manifest`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteKind {
    Flush,
    CompactionOutput,
    Manifest,
    Wal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteIntent {
    pub kind: WriteKind,
}

impl WriteIntent {
    pub fn new(kind: WriteKind) -> Self {
        Self { kind }
    }

    pub fn flush() -> Self {
        Self::new(WriteKind::Flush)
    }

    pub fn wal() -> Self {
        Self::new(WriteKind::Wal)
    }

    pub fn compaction_output() -> Self {
        Self::new(WriteKind::CompactionOutput)
    }

    pub fn manifest() -> Self {
        Self::new(WriteKind::Manifest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadKind {
    Foreground,
    CompactionInput,
    Warmup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryReason {
    CrcMismatch,
    BlockDecodeError,
    DecompressionError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadIntent {
    pub kind: ReadKind,
    pub retry: Option<RetryReason>,
}

impl ReadIntent {
    pub fn new(kind: ReadKind) -> Self {
        Self { kind, retry: None }
    }

    pub fn foreground() -> Self {
        Self::new(ReadKind::Foreground)
    }

    pub fn compaction_input() -> Self {
        Self::new(ReadKind::CompactionInput)
    }

    pub fn warmup() -> Self {
        Self::new(ReadKind::Warmup)
    }

    pub fn with_retry(mut self, reason: RetryReason) -> Self {
        self.retry = Some(reason);
        self
    }
}

/// Attach a `WriteIntent` to an existing `Extensions` map.
pub fn set_write_intent(extensions: &mut Extensions, intent: WriteIntent) {
    extensions.insert(intent);
}

/// Attach a `ReadIntent` to an existing `Extensions` map.
pub fn set_read_intent(extensions: &mut Extensions, intent: ReadIntent) {
    extensions.insert(intent);
}

/// Extract a previously attached `WriteIntent`, if any.
pub fn get_write_intent(extensions: &Extensions) -> Option<WriteIntent> {
    extensions.get::<WriteIntent>().copied()
}

/// Extract a previously attached `ReadIntent`, if any.
pub fn get_read_intent(extensions: &Extensions) -> Option<ReadIntent> {
    extensions.get::<ReadIntent>().copied()
}

/// Ergonomic constructors for the three options structs that carry
/// extensions. Each builds a fresh options struct and attaches the
/// intent in one place, so SlateDB call sites stay one line.
///
/// Allocation cost: one `Extensions` per call (`Box<HashMap>` + one
/// bucket + `Box<intent>`). Per RFC, this is intrinsic to the
/// `Extensions` API and cannot be amortized across calls (`GetOptions`
/// and friends are taken by value upstream).
pub fn get_options_with_intent(intent: ReadIntent) -> GetOptions {
    let mut opts = GetOptions::default();
    set_read_intent(&mut opts.extensions, intent);
    opts
}

pub fn head_options_with_intent(intent: ReadIntent) -> GetOptions {
    let mut opts = GetOptions::default().with_head(true);
    set_read_intent(&mut opts.extensions, intent);
    opts
}

pub fn range_options_with_intent(intent: ReadIntent, range: std::ops::Range<u64>) -> GetOptions {
    let mut opts = GetOptions::default().with_range(Some(range));
    set_read_intent(&mut opts.extensions, intent);
    opts
}

pub fn put_options_with_intent(intent: WriteIntent) -> PutOptions {
    let mut opts = PutOptions::default();
    set_write_intent(&mut opts.extensions, intent);
    opts
}

pub fn put_options_with_intent_and_mode(
    intent: WriteIntent,
    mode: object_store::PutMode,
) -> PutOptions {
    let mut opts = PutOptions::from(mode);
    set_write_intent(&mut opts.extensions, intent);
    opts
}

pub fn put_multipart_options_with_intent(intent: WriteIntent) -> PutMultipartOptions {
    let mut opts = PutMultipartOptions::default();
    set_write_intent(&mut opts.extensions, intent);
    opts
}

/// Test-only utilities for verifying that intents reach a wrapper. Gated
/// on `cfg(any(test, feature = "test-util"))` so external users who
/// enable the `test-util` feature can plug `IntentRecorderObjectStore`
/// into their own integration tests.
#[cfg(any(test, feature = "test-util"))]
pub mod testing {
    #![allow(clippy::disallowed_methods, clippy::disallowed_types)]

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use futures::StreamExt;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
    };

    use super::{get_read_intent, get_write_intent, ReadIntent, WriteIntent};

    /// Records one observed call against the wrapped ObjectStore.
    #[derive(Debug, Clone)]
    pub struct RecordedCall {
        pub method: &'static str,
        pub path: Path,
        pub read_intent: Option<ReadIntent>,
        pub write_intent: Option<WriteIntent>,
    }

    /// Wraps an inner `ObjectStore` and records the method name + path
    /// + extracted intent (if any) for every call. The MultipartUpload
    /// returned by `put_multipart_opts` is also wrapped so that
    /// `put_part` and `complete` are recorded, with no intent (the
    /// intent only attaches to the init call).
    pub struct IntentRecorderObjectStore {
        inner: Arc<dyn ObjectStore>,
        log: Arc<Mutex<Vec<RecordedCall>>>,
    }

    impl IntentRecorderObjectStore {
        /// Build a recorder wrapping `inner`. Returns the wrapper and a
        /// shared log handle the caller uses to inspect recorded calls.
        pub fn new(inner: Arc<dyn ObjectStore>) -> (Arc<Self>, Arc<Mutex<Vec<RecordedCall>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            let store = Arc::new(Self {
                inner,
                log: Arc::clone(&log),
            });
            (store, log)
        }

        fn record(
            &self,
            method: &'static str,
            path: &Path,
            read_intent: Option<ReadIntent>,
            write_intent: Option<WriteIntent>,
        ) {
            self.log.lock().unwrap().push(RecordedCall {
                method,
                path: path.clone(),
                read_intent,
                write_intent,
            });
        }
    }

    impl std::fmt::Display for IntentRecorderObjectStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "IntentRecorderObjectStore({})", self.inner)
        }
    }

    impl std::fmt::Debug for IntentRecorderObjectStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("IntentRecorderObjectStore").finish()
        }
    }

    struct RecordingMultipartUpload {
        inner: Box<dyn MultipartUpload>,
        log: Arc<Mutex<Vec<RecordedCall>>>,
        path: Path,
    }

    impl std::fmt::Debug for RecordingMultipartUpload {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RecordingMultipartUpload").finish()
        }
    }

    #[async_trait]
    impl MultipartUpload for RecordingMultipartUpload {
        fn put_part(&mut self, data: PutPayload) -> object_store::UploadPart {
            self.log.lock().unwrap().push(RecordedCall {
                method: "multipart_put_part",
                path: self.path.clone(),
                read_intent: None,
                write_intent: None,
            });
            self.inner.put_part(data)
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            self.log.lock().unwrap().push(RecordedCall {
                method: "multipart_complete",
                path: self.path.clone(),
                read_intent: None,
                write_intent: None,
            });
            self.inner.complete().await
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            self.inner.abort().await
        }
    }

    #[async_trait]
    impl ObjectStore for IntentRecorderObjectStore {
        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let intent = get_read_intent(&options.extensions);
            let method = if options.head {
                "get_opts[head]"
            } else if options.range.is_some() {
                "get_opts[range]"
            } else {
                "get_opts"
            };
            self.record(method, location, intent, None);
            self.inner.get_opts(location, options).await
        }

        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            let intent = get_write_intent(&opts.extensions);
            self.record("put_opts", location, None, intent);
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            let intent = get_write_intent(&opts.extensions);
            self.record("put_multipart_opts", location, None, intent);
            let inner = self.inner.put_multipart_opts(location, opts).await?;
            Ok(Box::new(RecordingMultipartUpload {
                inner,
                log: Arc::clone(&self.log),
                path: location.clone(),
            }))
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            let log = Arc::clone(&self.log);
            let inner = Arc::clone(&self.inner);
            locations
                .then(move |loc| {
                    let log = Arc::clone(&log);
                    let inner = Arc::clone(&inner);
                    async move {
                        let loc = loc?;
                        log.lock().unwrap().push(RecordedCall {
                            method: "delete",
                            path: loc.clone(),
                            read_intent: None,
                            write_intent: None,
                        });
                        inner.delete(&loc).await.map(|_| loc)
                    }
                })
                .boxed()
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.log.lock().unwrap().push(RecordedCall {
                method: "list",
                path: prefix.cloned().unwrap_or_else(|| Path::from("")),
                read_intent: None,
                write_intent: None,
            });
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.log.lock().unwrap().push(RecordedCall {
                method: "list_with_delimiter",
                path: prefix.cloned().unwrap_or_else(|| Path::from("")),
                read_intent: None,
                write_intent: None,
            });
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.record("copy_opts", from, None, None);
            self.inner.copy_opts(from, to, options).await
        }

        async fn rename_opts(
            &self,
            from: &Path,
            to: &Path,
            options: RenameOptions,
        ) -> object_store::Result<()> {
            self.record("rename_opts", from, None, None);
            self.inner.rename_opts(from, to, options).await
        }
    }
}

#[cfg(test)]
mod tests {
    //! Verification matrix: for every `ObjectStore` method shape that
    //! SlateDB uses today, drive a real call with an intent attached and
    //! assert whether the recorder wrapper observed it.
    //!
    //! The recorder forwards every call to an in-memory backend. Each
    //! test is named for what it proves, and the `EXPECTED:` comment on
    //! top of each test records the verdict that lands in the RFC.

    #![allow(clippy::disallowed_methods, clippy::disallowed_types)]

    use std::collections::VecDeque;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use futures::StreamExt;
    use object_store::buffered::BufWriter;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
    use tokio::io::AsyncWriteExt;

    use super::testing::{IntentRecorderObjectStore, RecordedCall};
    use super::*;

    fn make_recorder() -> (Arc<dyn ObjectStore>, Arc<Mutex<Vec<RecordedCall>>>) {
        let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let (store, log) = IntentRecorderObjectStore::new(backend);
        (store as Arc<dyn ObjectStore>, log)
    }

    fn first_with_method(log: &Mutex<Vec<RecordedCall>>, method: &str) -> Option<RecordedCall> {
        log.lock()
            .unwrap()
            .iter()
            .find(|c| c.method == method)
            .cloned()
    }

    fn all_methods(log: &Mutex<Vec<RecordedCall>>) -> VecDeque<&'static str> {
        log.lock().unwrap().iter().map(|c| c.method).collect()
    }

    // ============================================================
    // VERIFICATION MATRIX
    // ============================================================
    //
    // Each test exercises one ObjectStore method shape that SlateDB
    // uses, attaches an intent via the Extensions API, and asserts
    // whether the recorder observed it. The verdict is in the doc
    // comment on top of each test.

    /// EXPECTED PASS. `put_opts` is the canonical write path: SlateDB
    /// uses it directly for WAL writes (tablestore.rs:991). Extensions
    /// flow through unchanged.
    #[tokio::test]
    async fn put_opts_carries_write_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("wal/000001.sst");
        let mut opts = PutOptions::default();
        set_write_intent(&mut opts.extensions, WriteIntent::wal());

        store
            .put_opts(&path, PutPayload::from_static(b"hello"), opts)
            .await
            .unwrap();

        let call = first_with_method(&log, "put_opts").expect("put_opts not recorded");
        assert_eq!(
            call.write_intent,
            Some(WriteIntent::wal()),
            "put_opts must deliver the WriteIntent to the wrapper"
        );
    }

    /// EXPECTED PASS. Multipart init is what `BufWriter` falls back to
    /// for large writes. Extensions on `PutMultipartOptions` reach the
    /// wrapper.
    #[tokio::test]
    async fn put_multipart_opts_carries_write_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("compacted/000002.sst");
        let mut opts = PutMultipartOptions::default();
        set_write_intent(&mut opts.extensions, WriteIntent::compaction_output());

        let mut upload = store.put_multipart_opts(&path, opts).await.unwrap();
        upload
            .put_part(PutPayload::from(vec![0u8; 5 * 1024 * 1024]))
            .await
            .unwrap();
        upload.complete().await.unwrap();

        let call =
            first_with_method(&log, "put_multipart_opts").expect("put_multipart_opts not recorded");
        assert_eq!(call.write_intent, Some(WriteIntent::compaction_output()),);
    }

    /// EXPECTED PASS. `get_opts` is the canonical read path. SlateDB
    /// would tag every foreground / compaction-input / warmup read here.
    #[tokio::test]
    async fn get_opts_carries_read_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000003.sst");
        store
            .put(&path, PutPayload::from_static(b"abc"))
            .await
            .unwrap();

        let mut opts = GetOptions::default();
        set_read_intent(&mut opts.extensions, ReadIntent::foreground());
        store.get_opts(&path, opts).await.unwrap();

        let call = first_with_method(&log, "get_opts").expect("get_opts not recorded");
        assert_eq!(call.read_intent, Some(ReadIntent::foreground()));
    }

    /// EXPECTED PASS. Retry-after-validation-failure is a key contract:
    /// the wrapper must see `retry = Some(..)` so it evicts the
    /// corrupted cached file.
    #[tokio::test]
    async fn get_opts_carries_retry_read_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000004.sst");
        store
            .put(&path, PutPayload::from_static(b"abc"))
            .await
            .unwrap();

        let intent = ReadIntent::foreground().with_retry(RetryReason::CrcMismatch);
        let mut opts = GetOptions::default();
        set_read_intent(&mut opts.extensions, intent);
        store.get_opts(&path, opts).await.unwrap();

        let call = first_with_method(&log, "get_opts").unwrap();
        assert_eq!(call.read_intent, Some(intent));
        assert_eq!(
            call.read_intent.unwrap().retry,
            Some(RetryReason::CrcMismatch)
        );
    }

    /// EXPECTED PASS via range option on `get_opts`. SlateDB uses
    /// `ObjectStore::get_range` today (tablestore.rs:54), which routes
    /// through `get_opts` with the range option set, so intents attached
    /// to `GetOptions` flow naturally.
    #[tokio::test]
    async fn get_range_via_get_opts_carries_read_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000005.sst");
        store
            .put(&path, PutPayload::from_static(b"hello-world"))
            .await
            .unwrap();

        let opts = GetOptions::default()
            .with_range(Some(Range { start: 0, end: 5 }))
            .with_extensions({
                let mut ext = object_store::Extensions::new();
                set_read_intent(&mut ext, ReadIntent::compaction_input());
                ext
            });
        store.get_opts(&path, opts).await.unwrap();

        let call =
            first_with_method(&log, "get_opts[range]").expect("get_opts[range] not recorded");
        assert_eq!(call.read_intent, Some(ReadIntent::compaction_input()));
    }

    /// EXPECTED PASS via `get_opts(.., with_head(true)..)`. SlateDB
    /// currently calls `ObjectStore::head` (tablestore.rs:49, 453, 977),
    /// which has NO `_opts` variant in object_store 0.13. So intent can
    /// reach the wrapper only if SlateDB stops using `head` and routes
    /// the head request through `get_opts`. This test confirms that
    /// route works.
    #[tokio::test]
    async fn head_via_get_opts_carries_read_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000006.sst");
        store
            .put(&path, PutPayload::from_static(b"abc"))
            .await
            .unwrap();

        let opts = GetOptions::default().with_head(true).with_extensions({
            let mut ext = object_store::Extensions::new();
            set_read_intent(&mut ext, ReadIntent::foreground());
            ext
        });
        store.get_opts(&path, opts).await.unwrap();

        let call = first_with_method(&log, "get_opts[head]").expect("get_opts[head] not recorded");
        assert_eq!(call.read_intent, Some(ReadIntent::foreground()));
    }

    /// EXPECTED FAIL (documents a SlateDB-side change needed). The
    /// `ObjectStore::head` convenience routes through `get_opts` with
    /// default options, so intent CANNOT be attached. The wrapper sees a
    /// head call with no read_intent. Migrating SlateDB call sites that
    /// use `head()` to `get_opts(.., with_head(true).with_extensions(..))`
    /// is required.
    #[tokio::test]
    async fn head_convenience_drops_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000007.sst");
        store
            .put(&path, PutPayload::from_static(b"abc"))
            .await
            .unwrap();

        let _meta = store.head(&path).await.unwrap();

        let call = first_with_method(&log, "get_opts[head]").expect("get_opts[head] not recorded");
        assert!(
            call.read_intent.is_none(),
            "head() convenience cannot carry intent, so the wrapper sees None"
        );
    }

    /// EXPECTED FAIL (architectural gap in object_store 0.13). `delete`
    /// has no `delete_opts` variant, and `delete_stream` only carries
    /// paths in its stream. There is no Extensions slot to populate.
    /// The wrapper receives the delete with no intent attached.
    ///
    /// For RFC 0026, this is not a problem: the cache wrapper only needs
    /// to know that a delete happened so it can evict its local copy.
    /// No semantic intent is needed.
    #[tokio::test]
    async fn delete_cannot_carry_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("data/000008.sst");
        store
            .put(&path, PutPayload::from_static(b"abc"))
            .await
            .unwrap();

        store.delete(&path).await.unwrap();

        let call = first_with_method(&log, "delete").expect("delete not recorded");
        assert!(call.read_intent.is_none());
        assert!(call.write_intent.is_none());
    }

    /// EXPECTED FAIL (architectural gap in object_store 0.13). `list`
    /// takes only a prefix path. The paginated variant has options but
    /// is a different shape. SlateDB lists for WAL discovery and GC; the
    /// cache wrapper cannot distinguish these via Extensions today.
    #[tokio::test]
    async fn list_cannot_carry_intent() {
        let (store, log) = make_recorder();
        let _ = store
            .list(Some(&Path::from("data/")))
            .collect::<Vec<_>>()
            .await;

        let call = first_with_method(&log, "list").expect("list not recorded");
        assert!(call.read_intent.is_none());
        assert!(call.write_intent.is_none());
    }

    /// EXPECTED FAIL (upstream object_store 0.13 BufWriter bug). When
    /// the total payload fits in BufWriter's capacity, shutdown builds
    /// `PutOptions` with `..Default::default()`, dropping the extensions
    /// the BufWriter was constructed with. See
    /// arrow-rs-object-store/v0.13.2/src/buffered.rs:443-455.
    ///
    /// SlateDB uses BufWriter for L0 flush (after PR #1692) and
    /// compaction output (tablestore.rs:283, 1011). Any output smaller
    /// than ~10 MiB (the default capacity) reaches the cache wrapper
    /// UNTAGGED. This is the gap the RFC comment #13 flagged.
    #[tokio::test]
    async fn bufwriter_small_payload_drops_write_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("compacted/small.sst");

        let mut writer = BufWriter::new(Arc::clone(&store), path.clone()).with_extensions({
            let mut ext = object_store::Extensions::new();
            set_write_intent(&mut ext, WriteIntent::compaction_output());
            ext
        });
        writer.write_all(b"small payload").await.unwrap();
        writer.shutdown().await.unwrap();

        let methods = all_methods(&log);
        assert!(
            methods.iter().any(|m| *m == "put_opts"),
            "small BufWriter payload should route through put_opts on shutdown, got {:?}",
            methods
        );
        let put = first_with_method(&log, "put_opts").unwrap();
        assert!(
            put.write_intent.is_none(),
            "BUG: BufWriter drops extensions on the single-shot put_opts shutdown path. \
             Got {:?}",
            put.write_intent
        );
    }

    // ============================================================
    // FOCUSED TABLESTORE WRITE-PATH TEST
    // ============================================================
    //
    // Drives the production SST write path (`EncodedSsTableWriter` ->
    // `BufWriter` -> wrapper) with a small enough `BufWriter` capacity
    // that any meaningful payload overflows into the multipart branch
    // (where extensions ARE forwarded upstream). This proves SlateDB's
    // `new_intent_tagged_bufwriter` helper attaches extensions
    // correctly without depending on upstream issue #735 being fixed.

    /// Construct a `TableStore` whose underlying `BufWriter` uses a
    /// tiny capacity so SST writes always overflow into the multipart
    /// path. Drive an SST write end-to-end through the table-writer
    /// path and assert the wrapper saw `put_multipart_opts` with the
    /// requested `WriteIntent`.
    #[tokio::test]
    async fn tablestore_with_small_bufwriter_capacity_tags_writes_via_multipart() {
        use crate::db_state::SsTableId;
        use crate::format::sst::SsTableFormat;
        use crate::object_stores::ObjectStores;
        use crate::tablestore::TableStore;
        use crate::types::RowEntry;
        use object_store::path::Path as ObjPath;
        use ulid::Ulid;

        let (store, log) = make_recorder();
        let object_stores = ObjectStores::new(Arc::clone(&store), None);
        let ts = Arc::new(TableStore::new_with_bufwriter_capacity(
            object_stores,
            SsTableFormat::default(),
            ObjPath::from("/tmp/test_e2e_tablestore"),
            None,
            // 1 KB capacity: any non-empty SST overflows immediately,
            // forcing BufWriter onto the multipart branch where
            // extensions are correctly forwarded.
            1024,
        ));

        let id = SsTableId::Compacted(Ulid::new());
        let mut writer = ts.table_writer(id, WriteIntent::compaction_output());
        // A handful of entries is enough; the BufWriter overflows on
        // the first block flushed past 1 KB.
        for i in 0..64u32 {
            let key = format!("key{:08}", i);
            let value = vec![b'x'; 64];
            let entry = RowEntry::new_value(key.as_bytes(), &value, i as u64);
            writer.add(entry).await.unwrap();
        }
        writer.close().await.unwrap();

        let multipart_inits = intents_for_method(&log, "put_multipart_opts");
        assert_eq!(
            multipart_inits.len(),
            1,
            "expected exactly one put_multipart_opts init for the SST, saw {:?}",
            multipart_inits
        );
        let (_, write_intent) = multipart_inits[0];
        assert_eq!(
            write_intent,
            Some(WriteIntent::compaction_output()),
            "multipart init should carry the CompactionOutput intent"
        );

        // Bonus: no single-PUT path should have run at all, since
        // capacity = 1 KB forces overflow immediately.
        let single_puts = intents_for_method(&log, "put_opts");
        assert!(
            single_puts.is_empty(),
            "expected no put_opts (single-PUT path) but saw {:?}",
            single_puts
        );
    }

    // ============================================================
    // END-TO-END PRODUCTION-WIRING TESTS
    // ============================================================
    //
    // These drive a real `Db` whose object store is the recorder
    // wrapper, and assert that the expected intents arrive on
    // production paths. Note: small SST writes go through the
    // upstream `BufWriter` single-PUT path which drops extensions
    // (issue #735). The assertions below match that reality and
    // accept untagged small puts for now.

    fn intents_for_method(
        log: &Mutex<Vec<RecordedCall>>,
        method: &str,
    ) -> Vec<(Option<ReadIntent>, Option<WriteIntent>)> {
        log.lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .map(|c| (c.read_intent, c.write_intent))
            .collect()
    }

    /// Flush a WAL and assert at least one put_opts is `Wal`-tagged.
    ///
    /// We deliberately do NOT assert "no untagged puts" here: `db.close()`
    /// drains the memtable to an L0 SST, which flows through
    /// `BufWriter`. Small payloads hit the single-PUT shutdown branch
    /// that drops `Extensions` (upstream
    /// <https://github.com/apache/arrow-rs-object-store/issues/735>).
    /// Once that fix lands the strict "no untagged" assertion can be
    /// restored.
    #[tokio::test]
    async fn end_to_end_wal_flush_tags_writes_as_wal() {
        let (store, log) = make_recorder();
        let db = crate::Db::builder("/tmp/test_e2e_wal", store)
            .build()
            .await
            .unwrap();
        db.put(b"k", b"v").await.unwrap();
        db.flush().await.unwrap();
        db.close().await.unwrap();

        let puts = intents_for_method(&log, "put_opts");
        assert!(
            !puts.is_empty(),
            "expected at least one put_opts after WAL flush"
        );
        let wal_writes: Vec<_> = puts
            .iter()
            .filter(|(_, w)| *w == Some(WriteIntent::wal()))
            .collect();
        assert!(
            !wal_writes.is_empty(),
            "expected at least one Wal-tagged put_opts, saw {:?}",
            puts
        );
    }

    /// Forcing a memtable flush writes an L0 SST through `BufWriter`.
    /// Whether that put carries `WriteIntent::flush()` depends on which
    /// `BufWriter` branch it lands on:
    ///
    /// - small payload (single-PUT shutdown): extensions DROPPED upstream
    ///   (<https://github.com/apache/arrow-rs-object-store/issues/735>).
    /// - large payload (multipart overflow): extensions FORWARDED.
    ///
    /// Single-key test payloads fit in the 10 MiB default capacity, so
    /// today this test cannot assert the Flush tag is present. It
    /// instead verifies the flush still triggers at least one put_opts
    /// (i.e. the path is intact and we didn't accidentally skip the
    /// write). Once upstream #735 lands, tighten this to assert
    /// `WriteIntent::flush()` is present and no put is untagged.
    #[tokio::test]
    async fn end_to_end_memtable_flush_emits_put() {
        let (store, log) = make_recorder();
        let db = crate::Db::builder("/tmp/test_e2e_flush", store)
            .build()
            .await
            .unwrap();
        db.put(b"k", b"v").await.unwrap();
        db.flush_with_options(crate::config::FlushOptions {
            flush_type: crate::config::FlushType::MemTable,
        })
        .await
        .unwrap();
        db.close().await.unwrap();

        let puts = intents_for_method(&log, "put_opts");
        assert!(
            !puts.is_empty(),
            "expected at least one put_opts after memtable flush"
        );
    }

    /// Drive a real SST read and corrupt the bytes underneath the
    /// `ObjectStore` so the decoder fails CRC. The read path should
    /// re-issue the get_opts with `retry = Some(CrcMismatch)`, which
    /// the recorder observes. This proves the retry tag is actually
    /// emitted today (the wire was always there, but the reissue policy
    /// only landed with the `retry_reason_for` helper in `tablestore`).
    #[tokio::test]
    async fn end_to_end_crc_mismatch_emits_retry_tag() {
        use object_store::path::Path as ObjPath;

        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let (recorder, log) = IntentRecorderObjectStore::new(Arc::clone(&inner));
        let recorder_arc = recorder as Arc<dyn ObjectStore>;

        // Build a Db that actually writes some SSTs.
        let db = crate::Db::builder("/tmp/test_e2e_retry", Arc::clone(&recorder_arc))
            .build()
            .await
            .unwrap();
        for i in 0..16u32 {
            db.put(format!("k{:04}", i).as_bytes(), b"v").await.unwrap();
        }
        db.flush_with_options(crate::config::FlushOptions {
            flush_type: crate::config::FlushType::MemTable,
        })
        .await
        .unwrap();
        db.close().await.unwrap();

        // Find the L0 SST in the inner store, overwrite with garbage so
        // any subsequent decode fails CRC.
        let mut listing = inner.list(Some(&ObjPath::from("tmp/test_e2e_retry/compacted/")));
        let mut sst_path: Option<ObjPath> = None;
        while let Some(meta) = listing.next().await {
            let m = meta.unwrap();
            sst_path = Some(m.location);
            break;
        }
        let sst_path = sst_path.expect("expected at least one L0 SST");
        inner
            .put(&sst_path, PutPayload::from(vec![0xFFu8; 4096]))
            .await
            .unwrap();

        // Re-open and trigger a read that walks the SST. The decode
        // failure should re-issue the get_opts with the retry tag.
        let db = crate::Db::builder("/tmp/test_e2e_retry", Arc::clone(&recorder_arc))
            .build()
            .await
            .unwrap();
        log.lock().unwrap().clear();
        let _ = db.get(b"k0000").await;
        db.close().await.unwrap();

        let retry_reads: Vec<_> = log
            .lock()
            .unwrap()
            .iter()
            .filter_map(|c| c.read_intent.and_then(|i| i.retry.map(|r| (c.method, r))))
            .collect();
        assert!(
            !retry_reads.is_empty(),
            "expected at least one get_opts re-issued with retry tag after CRC failure, \
             saw method-intents={:?}",
            log.lock()
                .unwrap()
                .iter()
                .map(|c| (c.method, c.read_intent))
                .collect::<Vec<_>>()
        );
    }

    /// EXPECTED PASS. When the payload exceeds BufWriter capacity, it
    /// switches to put_multipart_opts and the extensions are forwarded
    /// (buffered.rs:341-345, 401-405). We force the overflow with a
    /// small capacity.
    #[tokio::test]
    async fn bufwriter_large_payload_carries_write_intent() {
        let (store, log) = make_recorder();
        let path = Path::from("compacted/large.sst");

        // Capacity = 5 MiB == object_store enforced minimum part size.
        // Writing more than that forces the multipart path.
        let mut writer =
            BufWriter::with_capacity(Arc::clone(&store), path.clone(), 5 * 1024 * 1024)
                .with_extensions({
                    let mut ext = object_store::Extensions::new();
                    set_write_intent(&mut ext, WriteIntent::compaction_output());
                    ext
                });
        let chunk = vec![0u8; 1024 * 1024];
        for _ in 0..8 {
            writer.write_all(&chunk).await.unwrap();
        }
        writer.shutdown().await.unwrap();

        let methods = all_methods(&log);
        assert!(
            methods.iter().any(|m| *m == "put_multipart_opts"),
            "large BufWriter payload should route through put_multipart_opts, got {:?}",
            methods
        );
        let init = first_with_method(&log, "put_multipart_opts").unwrap();
        assert_eq!(
            init.write_intent,
            Some(WriteIntent::compaction_output()),
            "BufWriter overflow path DOES carry extensions"
        );
    }
}
