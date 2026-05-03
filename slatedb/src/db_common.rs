use log::warn;
use parking_lot::RwLockWriteGuard;

use crate::db::DbInner;
use crate::db_state::DbState;
use crate::error::SlateDBError;
use crate::oracle::Oracle;
use crate::wal_replay::ReplayedMemtable;

pub(crate) const MAX_WAL_FLUSHES_BEFORE_L0_FLUSH: u64 = 4096;

/// Threshold above which `state.write()` acquisition + held-time gets
/// logged. Set on the high side (1 ms) since this is per-batch and we
/// don't want chatter; the dips we're hunting are tens of ms.
const SLOW_STATE_WRITE_LOCK_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(1);

impl DbInner {
    pub(crate) fn maybe_freeze_current_memtable(&self) -> Result<(), SlateDBError> {
        let wal_id = self.wal_buffer.recent_flushed_wal_id();

        // Double-check pattern: take `state.read()` first to evaluate
        // the freeze condition, escalate to `state.write()` only when
        // we actually need to freeze. parking_lot::RwLock is *fair*,
        // so a writer acquisition queues all incoming readers behind
        // it (writer-preference is what prevents writer starvation).
        // This function used to take `state.write()` unconditionally
        // on every `write_batch` call (50+/sec at typical workloads),
        // and even though the work inside the lock is cheap, the
        // queuing it caused was visible as multi-millisecond stalls
        // in `scan_prefix_by_recency`'s `state.read()` acquisition.
        //
        // The check itself is read-only: estimating encoded SST size
        // and looking at imm_memtable / replay_after_wal_id needs no
        // mutation. `freeze_memtable` is idempotent (returns early on
        // empty memtable), so even if another writer races ahead and
        // freezes the same memtable between our read-check and our
        // subsequent write, the second freeze is a no-op.
        let needs_freeze = {
            let guard = self.state.read();
            let meta = guard.memtable().metadata();
            let last_freeze_wal_id = guard
                .state()
                .imm_memtable
                .front()
                .map(|imm| imm.recent_flushed_wal_id())
                .unwrap_or(guard.state().core().replay_after_wal_id);
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

        // Slow path: actually freeze. Time the write acquisition +
        // work so the existing slow-state-write trace still surfaces
        // pathological cases (e.g. the freeze itself blocking on a
        // saturated flush queue).
        #[allow(clippy::disallowed_methods)]
        let acquire_start = tokio::time::Instant::now();
        let mut guard = self.state.write();
        let acquire_elapsed = acquire_start.elapsed();
        #[allow(clippy::disallowed_methods)]
        let work_start = tokio::time::Instant::now();
        let result = self.freeze_memtable(&mut guard, wal_id);
        let work_elapsed = work_start.elapsed();
        let total_micros = (acquire_elapsed + work_elapsed).as_micros() as u64;
        self.db_stats.state_write_acquisitions.increment(1);
        self.db_stats
            .state_write_held_micros_total
            .increment(total_micros);
        if acquire_elapsed > SLOW_STATE_WRITE_LOCK_THRESHOLD
            || work_elapsed > SLOW_STATE_WRITE_LOCK_THRESHOLD
        {
            warn!(
                "slow state.write() in maybe_freeze_current_memtable: acquire={:?}, work={:?}",
                acquire_elapsed, work_elapsed,
            );
        }
        result
    }

    pub(crate) fn freeze_current_memtable(&self) -> Result<(), SlateDBError> {
        let wal_id = self.wal_buffer.recent_flushed_wal_id();
        let mut guard = self.state.write();
        self.freeze_memtable(&mut guard, wal_id)
    }

    pub(crate) fn freeze_memtable(
        &self,
        guard: &mut RwLockWriteGuard<'_, DbState>,
        wal_id: u64,
    ) -> Result<(), SlateDBError> {
        if guard.memtable().is_empty() {
            return Ok(());
        }

        guard.freeze_memtable(wal_id);
        let _ = self.memtable_flusher().notify_memtable_frozen();
        Ok(())
    }

    pub(crate) fn replay_memtable(
        &self,
        replayed_memtable: ReplayedMemtable,
    ) -> Result<(), SlateDBError> {
        let mut guard = self.state.write();

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
        self.freeze_memtable(&mut guard, recent_flushed_wal_id)?;

        let last_wal = replayed_memtable.last_wal_id;
        guard.modify(|modifier| modifier.state.manifest.value.core.next_wal_sst_id = last_wal + 1);

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

        // replace the memtable
        guard.replace_memtable(replayed_memtable.table);
        let dirty_manifest = guard.state().manifest.clone();
        drop(guard);
        self.status_manager.report_manifest(dirty_manifest.into());
        Ok(())
    }
}
