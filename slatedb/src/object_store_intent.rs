//! Typed intents attached to object store calls (RFC-0027).
//!
//! SlateDB tags every opts-capable [`ObjectStore`](object_store::ObjectStore)
//! call it issues with a typed intent describing the caller's purpose: a
//! [`WriteIntent`] on writes and a [`ReadIntent`] on reads. The intent is
//! inserted into the call's [`Extensions`] (carried by
//! [`GetOptions`](object_store::GetOptions),
//! [`PutOptions`](object_store::PutOptions), and
//! [`PutMultipartOptions`](object_store::PutMultipartOptions)).
//!
//! Calls with no opts-capable variant (`delete`, `list`, and the convenience
//! `head`) carry no intent and flow through wrappers unchanged.

use object_store::Extensions;

/// Declares the purpose of a write issued by SlateDB.
///
/// Attached to the extensions of every opts-capable write (`put_opts` and
/// multipart upload initiation). A wrapper can use the [`WriteKind`] to decide
/// admission per kind, for example admitting flushes while skipping
/// short-lived WAL or metadata objects.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteIntent {
    /// The kind of write being performed.
    pub kind: WriteKind,
}

/// The kind of write being performed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteKind {
    /// A memtable flush producing an L0 SST.
    Flush,
    /// An SST produced by compaction.
    CompactionOutput,
    /// Transactional database metadata: manifests, compactions state, and
    /// the GC boundary file. These objects include mutable files (for
    /// example the GC boundary), so wrappers must not assume they are
    /// immutable. Distinct from SST-internal metadata sections (index,
    /// filters, stats), which travel as part of data files.
    TransactionalMetadata,
    /// A write-ahead log segment.
    Wal,
}

impl WriteIntent {
    /// Creates a write intent with the given kind.
    pub fn new(kind: WriteKind) -> Self {
        Self { kind }
    }

    /// A memtable flush write ([`WriteKind::Flush`]).
    pub fn flush() -> Self {
        Self::new(WriteKind::Flush)
    }

    /// A compaction output write ([`WriteKind::CompactionOutput`]).
    pub fn compaction_output() -> Self {
        Self::new(WriteKind::CompactionOutput)
    }

    /// A transactional metadata write ([`WriteKind::TransactionalMetadata`]).
    pub fn transactional_metadata() -> Self {
        Self::new(WriteKind::TransactionalMetadata)
    }

    /// A write-ahead log write ([`WriteKind::Wal`]).
    pub fn wal() -> Self {
        Self::new(WriteKind::Wal)
    }

    /// Returns a fresh [`Extensions`] map carrying this intent.
    pub fn into_extensions(self) -> Extensions {
        let mut extensions = Extensions::new();
        extensions.insert(self);
        extensions
    }

    /// Reads the intent back from an [`Extensions`] map, if present.
    pub fn from_extensions(extensions: &Extensions) -> Option<Self> {
        extensions.get::<Self>().copied()
    }
}

/// Declares the purpose of a read issued by SlateDB.
///
/// Attached to the extensions of every opts-capable read (`get_opts`,
/// including head-style requests issued via `get_opts` with `head` set). A
/// wrapper can use the [`ReadKind`] to apply different policies to foreground
/// queries, compaction-input scans, warmups, and metadata reads, and use
/// [`ReadIntent::retry`] to detect a reissued read after a validation failure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadIntent {
    /// The kind of read being performed.
    pub kind: ReadKind,
    /// Set when SlateDB reissues a read after the previous response failed
    /// validation (CRC mismatch, block decode error, decompression error).
    ///
    /// A caching wrapper should treat this as a signal that any locally
    /// cached bytes for the path may be corrupt: drop them and fetch fresh
    /// bytes from upstream. A wrapper that keeps serving the same corrupt
    /// bytes would make the retry pointless and surface the validation
    /// failure to the caller.
    pub retry: Option<RetryReason>,
}

/// The kind of read being performed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadKind {
    /// A read on the caller's critical path (point reads and scans).
    Foreground,
    /// A bulk read of compaction input SSTs.
    CompactionInput,
    /// A read of a write-ahead log segment: replay at open, WAL scans via
    /// [`WalReader`](crate::WalReader), and WAL existence probes. Segments
    /// are consumed once and deleted shortly after their contents reach L0,
    /// so caching wrappers should not admit them.
    Wal,
    /// A cache warmup read issued via
    /// [`DbCacheManagerOps::warm_sst`](crate::DbCacheManagerOps::warm_sst).
    Warmup,
    /// A read of transactional database metadata: manifests, compactions
    /// state, and the GC boundary file. These objects include mutable files
    /// (for example the GC boundary) and conditional reads, so caching
    /// wrappers should serve them from upstream. Distinct from SST-internal
    /// metadata sections (index, filters, stats), which travel as part of
    /// data files.
    TransactionalMetadata,
}

impl ReadIntent {
    /// Creates a read intent with the given kind and no retry reason.
    pub fn new(kind: ReadKind) -> Self {
        Self { kind, retry: None }
    }

    /// A foreground read ([`ReadKind::Foreground`]).
    pub fn foreground() -> Self {
        Self::new(ReadKind::Foreground)
    }

    /// A compaction input read ([`ReadKind::CompactionInput`]).
    pub fn compaction_input() -> Self {
        Self::new(ReadKind::CompactionInput)
    }

    /// A write-ahead log read ([`ReadKind::Wal`]).
    pub fn wal() -> Self {
        Self::new(ReadKind::Wal)
    }

    /// A cache warmup read ([`ReadKind::Warmup`]).
    pub fn warmup() -> Self {
        Self::new(ReadKind::Warmup)
    }

    /// A transactional metadata read ([`ReadKind::TransactionalMetadata`]).
    pub fn transactional_metadata() -> Self {
        Self::new(ReadKind::TransactionalMetadata)
    }

    /// Returns this intent marked as a retry for the given reason.
    pub fn with_retry(self, reason: RetryReason) -> Self {
        Self {
            retry: Some(reason),
            ..self
        }
    }

    /// Returns a fresh [`Extensions`] map carrying this intent.
    pub fn into_extensions(self) -> Extensions {
        let mut extensions = Extensions::new();
        extensions.insert(self);
        extensions
    }

    /// Reads the intent back from an [`Extensions`] map, if present.
    pub fn from_extensions(extensions: &Extensions) -> Option<Self> {
        extensions.get::<Self>().copied()
    }
}

/// Why a read is being reissued after a validation failure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryReason {
    /// The previous response failed an SST checksum validation.
    CrcMismatch,
    /// The previous response could not be decoded as a valid SST section.
    BlockDecodeError,
    /// The previous response could not be decompressed.
    DecompressionError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_round_trip_write_intent_through_extensions() {
        let extensions = WriteIntent::compaction_output().into_extensions();

        let intent = WriteIntent::from_extensions(&extensions);

        assert_eq!(intent, Some(WriteIntent::new(WriteKind::CompactionOutput)));
        assert_eq!(ReadIntent::from_extensions(&extensions), None);
    }

    #[test]
    fn should_round_trip_read_intent_through_extensions() {
        let extensions = ReadIntent::foreground()
            .with_retry(RetryReason::CrcMismatch)
            .into_extensions();

        let intent = ReadIntent::from_extensions(&extensions);

        assert_eq!(
            intent,
            Some(ReadIntent {
                kind: ReadKind::Foreground,
                retry: Some(RetryReason::CrcMismatch),
            })
        );
        assert_eq!(WriteIntent::from_extensions(&extensions), None);
    }

    #[test]
    fn should_survive_extensions_clone() {
        // Options structs are cloned per retry attempt by RetryingObjectStore;
        // intents must survive the clone.
        let extensions = ReadIntent::warmup().into_extensions();

        let cloned = extensions.clone();

        assert_eq!(
            ReadIntent::from_extensions(&cloned),
            Some(ReadIntent::warmup())
        );
    }

    #[test]
    fn should_construct_intents_with_expected_kinds() {
        assert_eq!(WriteIntent::flush().kind, WriteKind::Flush);
        assert_eq!(WriteIntent::wal().kind, WriteKind::Wal);
        assert_eq!(
            WriteIntent::transactional_metadata().kind,
            WriteKind::TransactionalMetadata
        );
        assert_eq!(
            WriteIntent::compaction_output().kind,
            WriteKind::CompactionOutput
        );
        assert_eq!(ReadIntent::foreground().kind, ReadKind::Foreground);
        assert_eq!(
            ReadIntent::compaction_input().kind,
            ReadKind::CompactionInput
        );
        assert_eq!(ReadIntent::warmup().kind, ReadKind::Warmup);
        assert_eq!(
            ReadIntent::transactional_metadata().kind,
            ReadKind::TransactionalMetadata
        );
        assert_eq!(ReadIntent::foreground().retry, None);
    }
}
