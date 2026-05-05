use log::warn;

use crate::db::DbInner;
use crate::error::SlateDBError;
use crate::oracle::Oracle;
use crate::wal_replay::ReplayedMemtable;

pub(crate) const MAX_WAL_FLUSHES_BEFORE_L0_FLUSH: u64 = 4096;

/// Threshold above which the writer-side `state.modify()` work + mutex
/// acquisition gets logged. Set on the high side (1 ms) since this is
/// per-batch and we don't want chatter; the dips we're hunting are
/// tens of ms.
const SLOW_STATE_WRITE_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(1);

impl DbInner {
    pub(crate) fn maybe_freeze_current_memtable(&self) -> Result<(), SlateDBError> {
        let wal_id = self.wal_buffer.recent_flushed_wal_id();

        // Double-check pattern: snapshot the state via a lock-free
        // `ArcSwap` load, evaluate the freeze condition, and only
        // escalate to a writer-mutex acquisition when we actually
        // need to freeze. `freeze_memtable` is idempotent (returns
        // early on empty memtable), so even if another writer races
        // ahead and freezes the same memtable between our load and
        // our subsequent `modify`, the second freeze is a no-op.
        let needs_freeze = {
            let cow = self.state.state();
            let meta = cow.memtable.table().metadata();
            let last_freeze_wal_id = cow
                .imm_memtable
                .front()
                .map(|imm| imm.recent_flushed_wal_id())
                .unwrap_or(cow.core().replay_after_wal_id);
            let l0_sst_size_est = self
                .table_store
                .estimate_encoded_size_compacted(meta.entry_num, meta.entries_size_in_bytes);
            let wal_id_gap = wal_id
                .checked_sub(last_freeze_wal_id)
                .ok_or_else(|| SlateDBError::InvalidDBState)?;
            wal_id_gap >= MAX_WAL_FLUSHES_BEFORE_L0_FLUSH
                || l0_sst_size_est >= self.settings.l0_sst_size_bytes
        };

        if !needs_freeze {
            return Ok(());
        }

        // Slow path: actually freeze. Time the writer-mutex
        // acquisition + work so pathological cases (e.g. the freeze
        // itself blocking on a saturated flush queue) still surface.
        #[allow(clippy::disallowed_methods)]
        let start = tokio::time::Instant::now();
        let result = self.freeze_memtable(wal_id);
        let elapsed = start.elapsed();
        self.db_stats.state_write_acquisitions.increment(1);
        self.db_stats
            .state_write_held_micros_total
            .increment(elapsed.as_micros() as u64);
        if elapsed > SLOW_STATE_WRITE_THRESHOLD {
            warn!(
                "slow state.modify() in maybe_freeze_current_memtable: total={:?}",
                elapsed,
            );
        }
        result
    }

    pub(crate) fn freeze_current_memtable(&self) -> Result<(), SlateDBError> {
        let wal_id = self.wal_buffer.recent_flushed_wal_id();
        self.freeze_memtable(wal_id)
    }

    pub(crate) fn freeze_memtable(&self, wal_id: u64) -> Result<(), SlateDBError> {
        // Cheap pre-check off the published snapshot. If the memtable
        // is empty we skip the writer mutex entirely; freeze_memtable
        // on the DbState side is also idempotent on an empty table,
        // so any race here is benign.
        if self.state.state().memtable.is_empty() {
            return Ok(());
        }

        self.state.freeze_memtable(wal_id);
        let _ = self.memtable_flusher().notify_memtable_frozen();
        Ok(())
    }

    pub(crate) fn replay_memtable(
        &self,
        replayed_memtable: ReplayedMemtable,
    ) -> Result<(), SlateDBError> {
        // a WAL might contain the data across multiple memtables. we can only consider
        // last_wal_id - 1 as the recent persisted wal id when the memtable is reconstructed.
        // or when we need to replay again, we might risks to lose some WAL entries.
        let recent_flushed_wal_id = if replayed_memtable.last_wal_id > 0 {
            replayed_memtable.last_wal_id - 1
        } else {
            0
        };
        // Keep recent_flushed_wal_id ahead of imm_memtable WAL IDs so
        // maybe_freeze_memtable's subtraction doesn't underflow.
        self.wal_buffer
            .advance_recent_flushed_wal_id(recent_flushed_wal_id);
        self.freeze_memtable(recent_flushed_wal_id)?;

        let last_wal = replayed_memtable.last_wal_id;

        // update seqs and clock
        // we know these won't move backwards (even though the replayed wal files might contain some
        // older rows) because the wal replay iterator ignores any entries with seq num lower than
        // l0_last_seq from the manifest
        assert!(self.oracle.last_seq() <= replayed_memtable.last_seq);
        self.oracle.advance_last_seq(replayed_memtable.last_seq);
        assert!(self.oracle.last_committed_seq() <= replayed_memtable.last_seq);
        self.oracle
            .advance_committed_seq(replayed_memtable.last_seq);
        self.mono_clock.set_last_tick(replayed_memtable.last_tick)?;

        // Bundle the next_wal_sst_id bump and the memtable replacement
        // into a single `modify` so a concurrent reader can't observe
        // the replaced memtable without the matching manifest update.
        let dirty_manifest = self.state.modify(|cow| {
            cow.manifest.value.core.next_wal_sst_id = last_wal + 1;
            assert!(cow.memtable.is_empty());
            cow.memtable = std::sync::Arc::new(replayed_memtable.table);
            cow.manifest.clone()
        });
        self.status_manager.report_manifest(dirty_manifest.into());
        Ok(())
    }
}
