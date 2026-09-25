//! Builds manifest views from durable compaction progress.

use std::collections::HashSet;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::ops::RangeBounds;

use ulid::Ulid;

use crate::bytes_range::BytesRange;
use crate::compactor_state::{Compaction, SourceId};
use crate::db_state::{SortedRun, SsTableId, SsTableView};
use crate::error::SlateDBError;
use crate::manifest::ManifestCore;

/// Output and key intervals that the worker fully stored.
pub(crate) struct IncrementalProgress {
    /// Key ranges whose input views can be replaced by stored output.
    intervals: Vec<BytesRange>,
    /// Stored output views clipped to those ranges, in key and sequence order.
    output: Vec<SsTableView>,
    /// All recorded output IDs, including SSTs outside the current durable intervals.
    output_ids: HashSet<SsTableId>,
}

impl IncrementalProgress {
    /// Builds ranges and output views from the job's stored progress.
    ///
    /// Unfinished ranges exclude the last output key because more versions can follow.
    /// Completed ranges include their full key range, even when they have no output.
    pub(crate) fn from_compaction(job: &Compaction) -> Result<Self, SlateDBError> {
        let mut intervals: Vec<BytesRange> = Vec::new();
        let mut output = Vec::new();
        let mut output_ids = HashSet::new();
        for sub_compaction in job.subcompactions() {
            output_ids.extend(sub_compaction.output_ssts().iter().map(|sst| sst.id));
            let interval = if sub_compaction.completed() {
                Some(sub_compaction.range().clone())
            } else if let Some(last) = sub_compaction.output_ssts().last() {
                // Output writers record the last key, and recovery preserves it.
                let boundary = last
                    .info
                    .last_entry
                    .clone()
                    .ok_or(SlateDBError::InvalidCompaction)?;
                // The boundary key can still have versions in later output SSTs.
                BytesRange::try_new(
                    sub_compaction.range().start_bound().cloned(),
                    Excluded(boundary),
                )
                .and_then(|prefix| prefix.intersect(sub_compaction.range()))
            } else {
                None
            };
            if let Some(interval) = interval {
                // Clip every SST so earlier files cannot expose versions of the boundary key.
                output.extend(sub_compaction.output_ssts().iter().filter_map(|sst| {
                    SsTableView::identity(sst.clone()).try_with_visible_range(interval.clone())
                }));
                // Join ranges such as [a, m) and [m, z) into [a, z) so input views
                // need only one subtraction. Keep ranges with a gap separate:
                // keys in that gap still need their input data until the output is ready.
                if let Some(joined) = intervals.last().and_then(|last| last.union(&interval)) {
                    *intervals.last_mut().expect("previous interval exists") = joined;
                } else {
                    intervals.push(interval);
                }
            }
        }
        Ok(Self {
            intervals,
            output,
            output_ids,
        })
    }

    /// Updates manifest to replace consumed input keys with stored output.
    ///
    /// Keeps input runs, even when empty, so recovery can still find them by ID.
    /// If a view splits, the first piece keeps its ID.
    /// Calls `new_view_id` for each additional piece.
    ///
    /// All inputs must be sorted runs. The oldest input run also holds the output.
    /// Updates only the in-memory state. The caller controls when to commit it.
    pub(crate) fn apply_to_manifest(
        &self,
        core: &mut ManifestCore,
        job: &Compaction,
        mut new_view_id: impl FnMut() -> Ulid,
    ) -> Result<(), SlateDBError> {
        let spec = job.spec();
        if spec.is_drain() || spec.has_l0_sources() || spec.sources().is_empty() {
            return Err(SlateDBError::InvalidCompaction);
        }
        let destination = spec.destination().ok_or(SlateDBError::InvalidCompaction)?;
        let sources: HashSet<u32> = spec
            .sources()
            .iter()
            .map(SourceId::unwrap_sorted_run)
            .collect();
        let tree = core
            .tree_for_segment_mut(spec.segment())
            .ok_or(SlateDBError::InvalidCompaction)?;
        if !sources
            .iter()
            .all(|id| tree.compacted.iter().any(|sr| sr.id == *id))
        {
            return Err(SlateDBError::InvalidCompaction);
        }
        if !sources.contains(&destination) {
            return Err(SlateDBError::InvalidCompaction);
        }
        for run in &mut tree.compacted {
            if !sources.contains(&run.id) {
                continue;
            }
            let mut views = Vec::new();
            for view in run.sst_views() {
                // Replace previously committed output using the full recorded output list.
                if self.output_ids.contains(&view.sst.id) {
                    continue;
                }
                let mut retained = vec![view.compacted_effective_range().clone()];
                for interval in &self.intervals {
                    retained = retained
                        .into_iter()
                        .flat_map(|range| subtract(&range, interval))
                        .collect();
                }
                for (index, range) in retained.into_iter().enumerate() {
                    let mut piece = if &range == view.compacted_effective_range() {
                        view.clone()
                    } else {
                        view.try_with_visible_range(range)
                            .expect("retained range intersects its SST")
                    };
                    if index > 0 {
                        piece.id = new_view_id();
                    }
                    views.push(piece);
                }
            }
            if run.id == destination {
                views.extend(self.output.clone());
                // Stable sorting preserves version order when output SSTs start at the same key.
                views.sort_by(|left, right| {
                    left.compacted_effective_range()
                        .comparable_start_bound()
                        .cmp(&right.compacted_effective_range().comparable_start_bound())
                });
            }
            *run = SortedRun::new(run.id, views);
        }
        Ok(())
    }
}

/// Returns the parts of `range` outside `removed`, in key order.
/// Converts an inclusive removed end to its successor so the retained view starts inclusively.
fn subtract(range: &BytesRange, removed: &BytesRange) -> Vec<BytesRange> {
    let Some(overlap) = range.intersect(removed) else {
        return vec![range.clone()];
    };
    let mut pieces = Vec::new();
    let before = match overlap.start_bound() {
        Included(key) => Some(Excluded(key.clone())),
        Excluded(key) => Some(Included(key.clone())),
        Unbounded => None,
    };
    if let Some(end) = before {
        pieces.extend(BytesRange::try_new(range.start_bound().cloned(), end));
    }
    let after = match overlap.end_bound() {
        Excluded(key) => Some(Included(key.clone())),
        Included(key) => {
            // A byte string followed by zero is its immediate successor.
            let mut next = key.to_vec();
            next.push(0);
            Some(Included(next.into()))
        }
        Unbounded => None,
    };
    if let Some(start) = after {
        pieces.extend(BytesRange::try_new(start, range.end_bound().cloned()));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compactor_state::{CompactionContext, CompactionSpec};
    use crate::db_state::{SsTableHandle, SsTableInfo};
    use crate::format::sst::SST_FORMAT_VERSION_LATEST;
    use crate::subcompaction::Subcompaction;
    use bytes::Bytes;
    use std::sync::Arc;

    fn view(first: &'static [u8], last: &'static [u8]) -> SsTableView {
        SsTableView::identity(SsTableHandle::new(
            Ulid::new().into(),
            SST_FORMAT_VERSION_LATEST,
            SsTableInfo {
                first_entry: Some(Bytes::from_static(first)),
                last_entry: Some(Bytes::from_static(last)),
                ..Default::default()
            },
        ))
    }

    #[test]
    fn subtract_preserves_edges_and_gaps() {
        let range = BytesRange::from_slice(b"a".as_slice()..b"z".as_slice());
        let removed = BytesRange::from_slice(b"g".as_slice()..b"m".as_slice());
        assert_eq!(
            subtract(&range, &removed),
            vec![
                BytesRange::from_slice(b"a".as_slice()..b"g".as_slice()),
                BytesRange::from_slice(b"m".as_slice()..b"z".as_slice()),
            ]
        );
        assert_eq!(subtract(&range, &BytesRange::unbounded()), vec![]);
        assert_eq!(
            subtract(&range, &BytesRange::from_slice(b"z".as_slice()..)),
            vec![range]
        );
    }

    #[test]
    fn incremental_views_preserve_ids_projections_and_empty_runs() {
        let original = view(b"a", b"z")
            .with_visible_range(BytesRange::from_slice(b"c".as_slice()..b"x".as_slice()));
        let released = view(b"g", b"j");
        let output = view(b"g", b"l");
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![
            SortedRun::new(2, [released]),
            SortedRun::new(1, [original.clone()]),
        ];
        let job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(2), SourceId::SortedRun(1)], 1),
        );
        let progress = IncrementalProgress {
            intervals: vec![BytesRange::from_slice(b"g".as_slice()..b"m".as_slice())],
            output_ids: HashSet::from([output.sst.id]),
            output: vec![output],
        };
        progress
            .apply_to_manifest(&mut core, &job, Ulid::new)
            .unwrap();
        assert_eq!(
            core.tree
                .compacted
                .iter()
                .map(|run| run.id)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(core.tree.compacted[0].sst_views().is_empty());
        let views = core.tree.compacted[1].sst_views();
        assert_eq!(views.len(), 3);
        assert_eq!(views[0].id, original.id);
        assert_ne!(views[2].id, original.id);
        assert_eq!(
            views[0].compacted_effective_range(),
            &BytesRange::from_slice(b"c".as_slice()..b"g".as_slice())
        );
        assert_eq!(
            views[2].compacted_effective_range(),
            &BytesRange::from_slice(b"m".as_slice()..b"x".as_slice())
        );
        let before = core.clone();
        progress
            .apply_to_manifest(&mut core, &job, Ulid::new)
            .unwrap();
        assert_eq!(core, before);
    }

    #[test]
    fn clone_projection_preserves_partial_input_gaps_and_boundary_versions() {
        use crate::manifest::{Manifest, ProjectionConfig};
        let original = view(b"a", b"z");
        let left = original.with_visible_range(BytesRange::from_slice(..b"g".as_slice()));
        let mut right = original.with_visible_range(BytesRange::from_slice(b"m".as_slice()..));
        right.id = Ulid::new();
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![
            SortedRun::new(3, [view(b"g", b"l")]),
            SortedRun::new(2, [left, right]),
            SortedRun::new(1, [view(b"m", b"m"), view(b"m", b"m"), view(b"z", b"z")]),
        ];
        let manifest = Manifest::initial(core);
        let projected = Manifest::projected(
            &manifest,
            &ProjectionConfig::from_global_range(BytesRange::from_slice(
                b"b".as_slice()..b"y".as_slice(),
            )),
        )
        .unwrap();
        let retained = &projected.core.tree.compacted[1];
        assert!(retained.sst_views().iter().all(|view| !view
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"h"))));
        let point = Manifest::projected(
            &manifest,
            &ProjectionConfig::from_global_range(BytesRange::from_slice(
                b"m".as_slice()..=b"m".as_slice(),
            )),
        )
        .unwrap();
        assert_eq!(
            point
                .core
                .tree
                .compacted
                .iter()
                .find(|run| run.id == 1)
                .unwrap()
                .sst_views()
                .len(),
            2
        );
    }

    #[test]
    fn durable_progress_requires_last_key_and_includes_empty_completion() {
        let mut output = view(b"a", b"m").sst;
        let make_job = |output| {
            Compaction::new(
                Ulid::new(),
                CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
            )
            .with_ctx(Some(CompactionContext::new(
                vec![
                    Subcompaction::new(BytesRange::from_slice(..b"z".as_slice()))
                        .with_output_ssts(vec![output]),
                    Subcompaction::new(BytesRange::from_slice(b"z".as_slice()..))
                        .with_status(crate::subcompaction::SubcompactionStatus::Completed),
                ],
                Some(0),
            )))
        };
        let progress = IncrementalProgress::from_compaction(&make_job(output.clone())).unwrap();
        assert_eq!(
            progress.intervals,
            vec![
                BytesRange::from_slice(..b"m".as_slice()),
                BytesRange::from_slice(b"z".as_slice()..),
            ]
        );
        assert_eq!(progress.output.len(), 1);
        output.info.last_entry = None;
        assert!(matches!(
            IncrementalProgress::from_compaction(&make_job(output)),
            Err(SlateDBError::InvalidCompaction)
        ));
    }

    #[test]
    fn durable_prefix_excludes_all_versions_of_the_boundary_key() {
        let first = view(b"a", b"m");
        let last = view(b"m", b"m");
        let job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_ctx(Some(CompactionContext::new(
            vec![Subcompaction::new(BytesRange::unbounded())
                .with_output_ssts(vec![first.sst, last.sst])],
            Some(0),
        )));
        let progress = IncrementalProgress::from_compaction(&job).unwrap();
        assert_eq!(progress.output.len(), 1);
        assert!(!progress.output[0]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));
    }

    #[test]
    fn destination_rebuild_preserves_output_order_and_expands_previous_views() {
        let input = view(b"a", b"z");
        let outputs = [view(b"g", b"g"), view(b"g", b"g"), view(b"g", b"m")];
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![SortedRun::new(1, [input.clone()])];
        let mut job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_ctx(Some(CompactionContext::new(
            vec![
                Subcompaction::new(BytesRange::from_slice(b"g".as_slice()..b"n".as_slice()))
                    .with_output_ssts(outputs.iter().map(|view| view.sst.clone()).collect()),
            ],
            Some(0),
        )));
        let progress = IncrementalProgress::from_compaction(&job).unwrap();
        progress
            .apply_to_manifest(&mut core, &job, Ulid::new)
            .unwrap();
        let views = core.tree.compacted[0].sst_views();
        assert_eq!(views.len(), 5);
        assert_eq!(views[0].id, input.id);
        assert_eq!(
            views[1..4]
                .iter()
                .map(|view| view.sst.id)
                .collect::<Vec<_>>(),
            outputs.iter().map(|view| view.sst.id).collect::<Vec<_>>()
        );
        let tail_id = views[4].id;
        assert!(!views[3]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));

        let mut ctx = job.ctx().unwrap().clone();
        ctx.mark_completed(0);
        job.set_ctx(Some(ctx));
        let completed = IncrementalProgress::from_compaction(&job).unwrap();
        completed
            .apply_to_manifest(&mut core, &job, Ulid::new)
            .unwrap();
        let views = core.tree.compacted[0].sst_views();
        assert_eq!(views.len(), 5);
        assert_eq!(views[4].id, tail_id);
        assert!(views[3]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));
        assert_eq!(
            views[4].compacted_effective_range().start_bound(),
            Included(&Bytes::from_static(b"n"))
        );
        let before = core.clone();
        completed
            .apply_to_manifest(&mut core, &job, || panic!("no new views expected"))
            .unwrap();
        assert_eq!(core, before);
    }
}
