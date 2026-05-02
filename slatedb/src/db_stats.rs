use slatedb_common::metrics::{CounterFn, GaugeFn, MetricsRecorderHelper};
use std::sync::Arc;

pub use crate::merge_operator::MERGE_OPERATOR_OPERANDS;

use crate::merge_operator::{
    MERGE_OPERATOR_FLUSH_PATH, MERGE_OPERATOR_OPERANDS_DESCRIPTION, MERGE_OPERATOR_PATH_LABEL,
    MERGE_OPERATOR_READ_PATH,
};

macro_rules! db_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.db.", $suffix)
    };
}

pub const REQUEST_COUNT: &str = db_stat_name!("request_count");
pub const WRITE_OPS: &str = db_stat_name!("write_ops");
pub const WRITE_BATCH_COUNT: &str = db_stat_name!("write_batch_count");
pub const BACKPRESSURE_COUNT: &str = db_stat_name!("backpressure_count");
pub const IMMUTABLE_MEMTABLE_FLUSHES: &str = db_stat_name!("immutable_memtable_flushes");
pub const WAL_BUFFER_FLUSHES: &str = db_stat_name!("wal_buffer_flushes");
pub const WAL_BUFFER_FLUSH_REQUESTS: &str = db_stat_name!("wal_buffer_flush_requests");
pub const WAL_BUFFER_ESTIMATED_BYTES: &str = db_stat_name!("wal_buffer_estimated_bytes");
pub const TOTAL_MEM_SIZE_BYTES: &str = db_stat_name!("total_mem_size_bytes");
pub const L0_SST_COUNT: &str = db_stat_name!("l0_sst_count");
pub const L0_FLUSH_BYTES: &str = db_stat_name!("l0_flush_bytes");
pub const SST_FILTER_FALSE_POSITIVE_COUNT: &str = db_stat_name!("sst_filter_false_positive_count");
pub const SST_FILTER_POSITIVE_COUNT: &str = db_stat_name!("sst_filter_positive_count");
pub const SST_FILTER_NEGATIVE_COUNT: &str = db_stat_name!("sst_filter_negative_count");
pub const MANIFEST_UPDATE_COUNT: &str = db_stat_name!("manifest_update_count");
pub const MANIFEST_UPDATE_MICROS_TOTAL: &str = db_stat_name!("manifest_update_micros_total");
pub const STATE_WRITE_HELD_MICROS_TOTAL: &str = db_stat_name!("state_write_held_micros_total");
pub const STATE_WRITE_ACQUISITIONS: &str = db_stat_name!("state_write_acquisitions");

/// Label key distinguishing filter metrics for point lookups from those for
/// prefix scans. Value is one of [`FILTER_KIND_POINT`] or
/// [`FILTER_KIND_PREFIX`].
pub const FILTER_KIND_LABEL: &str = "kind";
pub const FILTER_KIND_POINT: &str = "point";
pub const FILTER_KIND_PREFIX: &str = "prefix";

pub(crate) struct DbStatsInner {
    pub(crate) immutable_memtable_flushes: Arc<dyn CounterFn>,
    pub(crate) wal_buffer_estimated_bytes: Arc<dyn GaugeFn>,
    pub(crate) wal_buffer_flushes: Arc<dyn CounterFn>,
    pub(crate) wal_buffer_flush_requests: Arc<dyn CounterFn>,
    pub(crate) sst_filter_point_false_positives: Arc<dyn CounterFn>,
    pub(crate) sst_filter_point_positives: Arc<dyn CounterFn>,
    pub(crate) sst_filter_point_negatives: Arc<dyn CounterFn>,
    pub(crate) sst_filter_prefix_false_positives: Arc<dyn CounterFn>,
    pub(crate) sst_filter_prefix_positives: Arc<dyn CounterFn>,
    pub(crate) sst_filter_prefix_negatives: Arc<dyn CounterFn>,
    pub(crate) backpressure_count: Arc<dyn CounterFn>,
    pub(crate) get_requests: Arc<dyn CounterFn>,
    pub(crate) scan_requests: Arc<dyn CounterFn>,
    pub(crate) flush_requests: Arc<dyn CounterFn>,
    pub(crate) write_batch_count: Arc<dyn CounterFn>,
    pub(crate) write_ops: Arc<dyn CounterFn>,
    pub(crate) total_mem_size_bytes: Arc<dyn GaugeFn>,
    pub(crate) l0_sst_count: Arc<dyn GaugeFn>,
    pub(crate) l0_flush_bytes: Arc<dyn CounterFn>,
    pub(crate) merge_operator_read_operands: Arc<dyn CounterFn>,
    pub(crate) merge_operator_flush_operands: Arc<dyn CounterFn>,
    /// Number of completed manifest writes (compaction commits, L0
    /// flush commits, checkpoint commits). Pair with
    /// [`Self::manifest_update_micros_total`] to compute average
    /// manifest commit duration per second.
    pub(crate) manifest_update_count: Arc<dyn CounterFn>,
    /// Cumulative wall-clock microseconds spent inside
    /// `manifest.update().await`, summed across all callers. Manifest
    /// writes are *not* held under the `state` write lock, but they
    /// can briefly stall progress on the writer/compactor that issued
    /// them. Per-second delta is the total time the compactor /
    /// memtable flusher spends blocked on manifest S3 PUTs.
    pub(crate) manifest_update_micros_total: Arc<dyn CounterFn>,
    /// Number of times `state.write()` was acquired anywhere in
    /// slatedb (memtable rotation, manifest application, etc.). Pair
    /// with [`Self::state_write_held_micros_total`] to chart average
    /// hold duration per second.
    pub(crate) state_write_acquisitions: Arc<dyn CounterFn>,
    /// Cumulative microseconds the `state` write lock was held.
    /// While held, every reader that calls `state.read()` blocks. A
    /// per-second delta in the same window as a read-throughput dip
    /// is direct evidence that lock contention is the cause.
    pub(crate) state_write_held_micros_total: Arc<dyn CounterFn>,
}

#[derive(Clone)]
pub(crate) struct DbStats {
    inner: Arc<DbStatsInner>,
}

impl std::ops::Deref for DbStats {
    type Target = DbStatsInner;

    #[inline]
    fn deref(&self) -> &DbStatsInner {
        &self.inner
    }
}

impl DbStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> DbStats {
        let inner = DbStatsInner {
            immutable_memtable_flushes: recorder.counter(IMMUTABLE_MEMTABLE_FLUSHES).register(),
            wal_buffer_estimated_bytes: recorder.gauge(WAL_BUFFER_ESTIMATED_BYTES).register(),
            wal_buffer_flushes: recorder.counter(WAL_BUFFER_FLUSHES).register(),
            wal_buffer_flush_requests: recorder.counter(WAL_BUFFER_FLUSH_REQUESTS).register(),
            sst_filter_point_false_positives: recorder
                .counter(SST_FILTER_FALSE_POSITIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_POINT)])
                .register(),
            sst_filter_point_positives: recorder
                .counter(SST_FILTER_POSITIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_POINT)])
                .register(),
            sst_filter_point_negatives: recorder
                .counter(SST_FILTER_NEGATIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_POINT)])
                .register(),
            sst_filter_prefix_false_positives: recorder
                .counter(SST_FILTER_FALSE_POSITIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_PREFIX)])
                .register(),
            sst_filter_prefix_positives: recorder
                .counter(SST_FILTER_POSITIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_PREFIX)])
                .register(),
            sst_filter_prefix_negatives: recorder
                .counter(SST_FILTER_NEGATIVE_COUNT)
                .labels(&[(FILTER_KIND_LABEL, FILTER_KIND_PREFIX)])
                .register(),
            backpressure_count: recorder.counter(BACKPRESSURE_COUNT).register(),
            get_requests: recorder
                .counter(REQUEST_COUNT)
                .labels(&[("op", "get")])
                .register(),
            scan_requests: recorder
                .counter(REQUEST_COUNT)
                .labels(&[("op", "scan")])
                .register(),
            flush_requests: recorder
                .counter(REQUEST_COUNT)
                .labels(&[("op", "flush")])
                .register(),
            write_batch_count: recorder.counter(WRITE_BATCH_COUNT).register(),
            write_ops: recorder.counter(WRITE_OPS).register(),
            total_mem_size_bytes: recorder.gauge(TOTAL_MEM_SIZE_BYTES).register(),
            l0_sst_count: recorder.gauge(L0_SST_COUNT).register(),
            l0_flush_bytes: recorder.counter(L0_FLUSH_BYTES).register(),
            merge_operator_read_operands: recorder
                .counter(MERGE_OPERATOR_OPERANDS)
                .labels(&[(MERGE_OPERATOR_PATH_LABEL, MERGE_OPERATOR_READ_PATH)])
                .description(MERGE_OPERATOR_OPERANDS_DESCRIPTION)
                .register(),
            merge_operator_flush_operands: recorder
                .counter(MERGE_OPERATOR_OPERANDS)
                .labels(&[(MERGE_OPERATOR_PATH_LABEL, MERGE_OPERATOR_FLUSH_PATH)])
                .description(MERGE_OPERATOR_OPERANDS_DESCRIPTION)
                .register(),
            manifest_update_count: recorder.counter(MANIFEST_UPDATE_COUNT).register(),
            manifest_update_micros_total: recorder
                .counter(MANIFEST_UPDATE_MICROS_TOTAL)
                .register(),
            state_write_acquisitions: recorder.counter(STATE_WRITE_ACQUISITIONS).register(),
            state_write_held_micros_total: recorder
                .counter(STATE_WRITE_HELD_MICROS_TOTAL)
                .register(),
        };
        DbStats {
            inner: Arc::new(inner),
        }
    }
}
