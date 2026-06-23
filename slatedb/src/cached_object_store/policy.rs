//! Per-call admission policy for the bundled object store cache.
//!
//! The decisions are pure functions of the call's
//! [`ObjectStoreCallTag`](crate::object_store_tag::ObjectStoreCallTag) (set by
//! the `TableStore`, read here) and the configured [`CachePutPolicy`]. Keeping
//! them here, free of any I/O, makes the full policy table exhaustively
//! matchable and unit-testable without touching the filesystem. See RFC 0027.

use crate::db_state::SstType;
use crate::object_store_tag::{ObjectStoreCallTag, TableStoreKind};

/// What the cache should do for a GET, decided from the call tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GetAction {
    /// Skip the cache entirely: no lookup, no admission. Used for compaction
    /// input scans (one-shot, should not pollute the cache) and for WAL reads
    /// (WAL is short-lived and GC'd after flush, so it is never cached).
    Bypass,
    /// Drop any cached entry for the path, then take the normal path so the
    /// read refetches from upstream and re-caches. Used when a read is reissued
    /// after a validation failure, so a corrupt local part is replaced rather
    /// than served again.
    Refetch,
    /// Normal cached read: serve from cache, admitting parts on a miss.
    Admit,
}

/// What the cache should do for a PUT, decided from the call tag and config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PutAction {
    /// Write through to upstream and also save the payload to the local cache.
    Cache,
    /// Write through to upstream only; leave the local cache untouched.
    Skip,
}

/// Which compacted SST write sources the cache admits. Both default to `false`,
/// preserving the historical write through disabled behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CachePutPolicy {
    /// Admit compacted SSTs written by a memtable flush (the main store).
    pub(crate) cache_on_flush: bool,
    /// Admit compacted SSTs written by compaction (the compactor store).
    pub(crate) cache_on_compaction: bool,
}

/// Decides the GET action for a data read from the call tag.
///
/// This applies to data GETs only. HEAD requests are handled by the caller
/// (`CachedObjectStore::get_opts`) before this is consulted: a non-WAL HEAD
/// always reads through the cache (cheap metadata that admits no data blocks, so
/// even compaction reads benefit from a cached head), while a WAL HEAD bypasses
/// the cache like its data reads.
///
/// Compactor data reads and WAL reads bypass the cache (compaction input scans
/// are one-shot; WAL is never cached). A reissued (retry) read refetches; every
/// other read (main, reader, GC, or untagged coordination I/O) takes the normal
/// admit on miss path. Bypass is checked first, so a WAL or compactor read that
/// is reissued after a validation failure still bypasses (there is nothing
/// cached to drop).
pub(crate) fn get_action(tag: Option<&ObjectStoreCallTag>) -> GetAction {
    match tag {
        Some(t) if t.kind == TableStoreKind::Compactor || t.sst_type == SstType::Wal => {
            GetAction::Bypass
        }
        Some(t) if t.retry.is_some() => GetAction::Refetch,
        _ => GetAction::Admit,
    }
}

/// Decides the PUT action from the call tag and the configured policy.
///
/// WAL writes are never cached (short-lived). A compacted SST write is cached
/// only when its source is enabled: `cache_on_flush` for the main store
/// (memtable flush), `cache_on_compaction` for the compactor. Untagged writes
/// (manifest, compaction state) are coordination I/O, not SST bytes, so they are
/// never cached here.
///
/// This is the decision for both single PUTs and multipart uploads
/// (`put_multipart_opts` calls it at init, where the tag is intact).
pub(crate) fn put_action(tag: Option<&ObjectStoreCallTag>, policy: &CachePutPolicy) -> PutAction {
    let Some(tag) = tag else {
        return PutAction::Skip;
    };
    match tag.sst_type {
        SstType::Wal => PutAction::Skip,
        SstType::Compacted => match tag.kind {
            TableStoreKind::Main if policy.cache_on_flush => PutAction::Cache,
            TableStoreKind::Compactor if policy.cache_on_compaction => PutAction::Cache,
            _ => PutAction::Skip,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RetryReason;
    use rstest::rstest;

    fn tag(
        kind: TableStoreKind,
        sst_type: SstType,
        retry: Option<RetryReason>,
    ) -> ObjectStoreCallTag {
        ObjectStoreCallTag {
            kind,
            sst_type,
            retry,
        }
    }

    #[rstest]
    // Compactor reads bypass regardless of sst type or retry.
    #[case(
        Some(tag(TableStoreKind::Compactor, SstType::Compacted, None)),
        GetAction::Bypass
    )]
    #[case(
        Some(tag(
            TableStoreKind::Compactor,
            SstType::Compacted,
            Some(RetryReason::CrcMismatch)
        )),
        GetAction::Bypass
    )]
    // WAL reads bypass for any source, even when reissued (WAL is never cached).
    #[case(Some(tag(TableStoreKind::Main, SstType::Wal, None)), GetAction::Bypass)]
    #[case(
        Some(tag(TableStoreKind::Reader, SstType::Wal, None)),
        GetAction::Bypass
    )]
    #[case(
        Some(tag(
            TableStoreKind::Main,
            SstType::Wal,
            Some(RetryReason::BlockDecodeError)
        )),
        GetAction::Bypass
    )]
    // A reissued, non-bypassed read refetches.
    #[case(
        Some(tag(
            TableStoreKind::Main,
            SstType::Compacted,
            Some(RetryReason::CrcMismatch)
        )),
        GetAction::Refetch
    )]
    #[case(
        Some(tag(
            TableStoreKind::Reader,
            SstType::Compacted,
            Some(RetryReason::BlockDecodeError)
        )),
        GetAction::Refetch
    )]
    // Everything else is a normal admit, including untagged coordination reads.
    #[case(
        Some(tag(TableStoreKind::Main, SstType::Compacted, None)),
        GetAction::Admit
    )]
    #[case(
        Some(tag(TableStoreKind::Reader, SstType::Compacted, None)),
        GetAction::Admit
    )]
    #[case(
        Some(tag(TableStoreKind::GC, SstType::Compacted, None)),
        GetAction::Admit
    )]
    #[case(None, GetAction::Admit)]
    fn test_get_action(#[case] tag: Option<ObjectStoreCallTag>, #[case] expected: GetAction) {
        assert_eq!(get_action(tag.as_ref()), expected);
    }

    #[rstest]
    // WAL writes are never cached, even with both flags on.
    #[case(
        Some(tag(TableStoreKind::Main, SstType::Wal, None)),
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: true },
        PutAction::Skip
    )]
    // Untagged writes (manifest, compaction state) are never cached.
    #[case(
        None,
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: true },
        PutAction::Skip
    )]
    // Flush writes (main store, compacted) gated by cache_on_flush.
    #[case(
        Some(tag(TableStoreKind::Main, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: false },
        PutAction::Cache
    )]
    #[case(
        Some(tag(TableStoreKind::Main, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: false, cache_on_compaction: true },
        PutAction::Skip
    )]
    // Compaction writes (compactor store, compacted) gated by cache_on_compaction.
    #[case(
        Some(tag(TableStoreKind::Compactor, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: false, cache_on_compaction: true },
        PutAction::Cache
    )]
    #[case(
        Some(tag(TableStoreKind::Compactor, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: false },
        PutAction::Skip
    )]
    // Reader/GC never write compacted SSTs, but if they did the policy is Skip.
    #[case(
        Some(tag(TableStoreKind::Reader, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: true },
        PutAction::Skip
    )]
    #[case(
        Some(tag(TableStoreKind::GC, SstType::Compacted, None)),
        CachePutPolicy { cache_on_flush: true, cache_on_compaction: true },
        PutAction::Skip
    )]
    fn test_put_action(
        #[case] tag: Option<ObjectStoreCallTag>,
        #[case] policy: CachePutPolicy,
        #[case] expected: PutAction,
    ) {
        assert_eq!(put_action(tag.as_ref(), &policy), expected);
    }

    #[test]
    fn test_default_put_policy_caches_nothing() {
        let policy = CachePutPolicy::default();
        assert!(!policy.cache_on_flush);
        assert!(!policy.cache_on_compaction);
        for kind in [
            TableStoreKind::Main,
            TableStoreKind::Compactor,
            TableStoreKind::Reader,
            TableStoreKind::GC,
        ] {
            assert_eq!(
                put_action(Some(&tag(kind, SstType::Compacted, None)), &policy),
                PutAction::Skip
            );
        }
    }
}
