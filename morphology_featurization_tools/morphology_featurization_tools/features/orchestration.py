"""
FeatureOrchestrator: run a set of feature extractors over a batch of fields
and return one row per grouping unit (well) of the batch.

  * crop-based per-cell features (structure_tensor, intensity_stats) take
    fixed-canvas crops from CropSegmenter;
  * field+mask per-cell features (cp_*, structure_tensor_field) take fields and
    label masks from FieldMaskSegmenter;
  * per-field features (tda) take whole fields.

Cellpose runs once per batch and both per-cell paths share its masks. Per-cell
values are aggregated to the grouping unit with aggregator_fn (default:
elementwise median over the batch's cells of that unit), per-field values over
the unit's fields, and the blocks are joined on the grouping keys. Aggregation
is per call, so a well whose fields span two calls gets two rows, which the
caller must combine.
"""

from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from .base import Regime, FeatureExtractor
from .field_mask import FieldMaskFeatureExtractor
from ..segmentation import CropSegmenter, FieldMaskSegmenter, FeatureAggregator


def _is_field_mask(extractor) -> bool:
    return isinstance(extractor, FieldMaskFeatureExtractor)


def _is_crop_per_cell(extractor) -> bool:
    return extractor.regime is Regime.PER_CELL and not _is_field_mask(extractor)


def _is_per_field(extractor) -> bool:
    return extractor.regime is Regime.PER_FIELD


def _measure_one(extractor, prepared, mask):
    """Measure one prepared field with one extractor (module level so joblib
    can pickle it). Returns the raw (dict, n_labels) result."""
    return extractor.measure_prepared(prepared, mask)


class FeatureOrchestrator:
    """
    Run a set of named feature extractors over a batch of fields and produce a
    concatenated, well-level profile.

    Parameters
    ----------
    extractors : list of FeatureExtractor
        Built via build_features([...]).
    grouping_keys : list of str
        Metadata columns defining the aggregation unit (e.g.
        ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]).
    crop_segmenter : CropSegmenter or None
        Segmenter producing fixed-canvas crops. Required if any crop-based
        per-cell features are present.
    field_mask_segmenter : FieldMaskSegmenter or None
        Segmenter producing fields + label masks. Required if any cp_measure
        features are present.
    aggregator_fn : callable or None
        Per-group cell aggregation function (default elementwise median).
    n_jobs : int
        Parallel workers for per-cell and field+mask extraction.
    report_timing : bool
        Print per-phase wall times for each batch.
    max_cells_per_pass : int
        Crop-based features process at most this many cells at a time.
    """

    def __init__(
        self,
        extractors: list[FeatureExtractor],
        grouping_keys: list[str],
        crop_segmenter: CropSegmenter | None = None,
        field_mask_segmenter: FieldMaskSegmenter | None = None,
        aggregator_fn=None,
        n_jobs: int = 1,
        report_timing: bool = False,
        max_cells_per_pass: int = 4096,
    ):
        self.report_timing = report_timing
        self.max_cells_per_pass = max_cells_per_pass
        self.extractors = extractors
        self.grouping_keys = grouping_keys
        self.crop_segmenter = crop_segmenter
        self.field_mask_segmenter = field_mask_segmenter
        self.aggregator_fn = aggregator_fn
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------

    def _aggregate(self, features: np.ndarray, metadata: pd.DataFrame):
        aggregator = FeatureAggregator(
            grouping_keys=self.grouping_keys, aggregator_fn=self.aggregator_fn
        )
        return aggregator.aggregate(features, metadata)

    def _block_to_frame(self, values, labels, columns):
        """Build a well-level DataFrame: grouping keys + named feature columns."""
        frame = labels.copy().reset_index(drop=True)
        feat = pd.DataFrame(values, columns=columns)
        return pd.concat([frame, feat], axis=1)

    # ------------------------------------------------------------------

    @staticmethod
    def _plan_passes(cell_counts: list[int], max_cells: int) -> list[list[int]]:
        """Group field indices into passes of at most `max_cells` cells (a
        field with more cells than that gets a pass of its own)."""
        if max_cells is None or max_cells <= 0:
            return [list(range(len(cell_counts)))]

        passes: list[list[int]] = []
        current: list[int] = []
        current_cells = 0
        for index, count in enumerate(cell_counts):
            if current and current_cells + count > max_cells:
                passes.append(current)
                current, current_cells = [], 0
            current.append(index)
            current_cells += count
        if current:
            passes.append(current)
        return passes or [[]]

    def run(self, batch: list[dict]) -> pd.DataFrame:
        """
        Featurize one batch of per-sample field dicts and return a well-level
        profile DataFrame (grouping keys + all requested feature columns),
        one row per unique grouping-key combination present in the batch.
        """
        crop_features = [e for e in self.extractors if _is_crop_per_cell(e)]
        fieldmask_features = [e for e in self.extractors if _is_field_mask(e)]
        field_features = [e for e in self.extractors if _is_per_field(e)]

        _timing: dict[str, float] = {}
        _debug_timing = self.report_timing

        blocks: list[pd.DataFrame] = []

        # Segment once; both per-cell paths use these masks.
        shared_masks = None
        if crop_features or fieldmask_features:
            mask_source = (
                self.crop_segmenter if crop_features else self.field_mask_segmenter
            )
            if mask_source is not None:
                _t = time.perf_counter()
                shared_masks = mask_source.segment_masks(batch)
                _timing["segment"] = time.perf_counter() - _t

        # Crop-based per-cell features
        if crop_features:
            if self.crop_segmenter is None:
                raise ValueError("crop_segmenter required for crop-based features.")

            # The crop stack grows with the number of cells (about 256 KiB per
            # cell at two channels and a 256 canvas), so crops are made in
            # passes of at most max_cells_per_pass cells. Only feature values
            # are kept between passes; aggregation runs once at the end.
            counts = (
                [int(np.asarray(m).max()) for m in shared_masks]
                if shared_masks is not None else [0] * len(batch)
            )
            passes = self._plan_passes(counts, self.max_cells_per_pass)

            values_by_feature: dict[str, list[np.ndarray]] = {
                extractor.name: [] for extractor in crop_features
            }
            meta_parts: list[pd.DataFrame] = []
            n_cells_total = 0

            for indices in passes:
                sub_batch = [batch[i] for i in indices]
                sub_masks = ([shared_masks[i] for i in indices]
                             if shared_masks is not None else None)
                _t = time.perf_counter()
                crops, cell_meta = self.crop_segmenter.segment(
                    sub_batch, masks=sub_masks
                )
                _timing["crop"] = _timing.get("crop", 0.0) + (time.perf_counter() - _t)
                if len(crops) == 0:
                    continue
                n_cells_total += len(crops)
                meta_parts.append(cell_meta)
                for extractor in crop_features:
                    _t = time.perf_counter()
                    values_by_feature[extractor.name].append(
                        extractor.extract(crops, n_jobs=self.n_jobs)
                    )
                    _timing[extractor.name] = (
                        _timing.get(extractor.name, 0.0) + (time.perf_counter() - _t)
                    )
                del crops

            _timing["n_cells"] = n_cells_total
            _timing["passes"] = len(passes)

            if meta_parts:
                cell_meta = pd.concat(meta_parts, ignore_index=True)
                for extractor in crop_features:
                    values = np.concatenate(values_by_feature[extractor.name], axis=0)
                    agg, labels = self._aggregate(values, cell_meta)
                    blocks.append(
                        self._block_to_frame(agg, labels, extractor.feature_names())
                    )

        # Field+mask per-cell features
        if fieldmask_features:
            if self.field_mask_segmenter is None:
                raise ValueError(
                    "field_mask_segmenter required for cp_measure features."
                )
            _t = time.perf_counter()
            fields, masks, field_meta = self.field_mask_segmenter.segment_fields(
                batch, masks=shared_masks
            )
            _timing["fields"] = time.perf_counter() - _t
            # One task per (extractor, field) pair, each carrying one
            # field-channel. Splitting by extractor instead would limit the
            # speedup to total cost / slowest extractor (~8.7x for the full
            # CellProfiler set).
            _t = time.perf_counter()
            prepared = [extractor.prepare(fields) for extractor in fieldmask_features]
            tasks = [(e, f) for e in range(len(fieldmask_features))
                     for f in range(len(fields))]
            flat = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_measure_one)(fieldmask_features[e], prepared[e][f], masks[f])
                for e, f in tasks
            )
            del prepared
            _timing["fieldmask_measure"] = time.perf_counter() - _t

            # Regroup by extractor and assemble here, in the parent.
            n_fields_here = len(fields)
            extracted = []
            for e, extractor in enumerate(fieldmask_features):
                per_field = flat[e * n_fields_here:(e + 1) * n_fields_here]
                values, provenance = extractor.assemble(per_field)
                extracted.append((values, provenance, extractor.feature_names()))
            del flat

            for extractor, (values, provenance, names) in zip(fieldmask_features,
                                                              extracted):
                extractor._names = names
                # Per-cell metadata from each row's field index.
                if len(provenance) == 0:
                    continue
                field_indices = [p[0] for p in provenance]
                cell_meta_fm = field_meta.iloc[field_indices].reset_index(drop=True)
                agg, labels = self._aggregate(values, cell_meta_fm)
                blocks.append(self._block_to_frame(agg, labels, extractor.feature_names()))

        # Per-field features
        if field_features:
            fields = [sample["image"] for sample in batch]
            field_meta = pd.DataFrame(
                [{k: v for k, v in sample.items() if k != "image"} for sample in batch]
            )
            for extractor in field_features:
                _t = time.perf_counter()
                if hasattr(extractor, "extract_fields"):
                    values, meta_out = extractor.extract_fields(
                        fields, field_meta, n_jobs=self.n_jobs
                    )
                else:  # pragma: no cover - autoencoder uses dataset path
                    raise ValueError(
                        f"Per-field extractor {extractor.name} needs a dataset; "
                        "use its extract_fields_from_dataset in a dedicated run."
                    )
                _timing[extractor.name] = time.perf_counter() - _t
                agg, labels = self._aggregate(np.asarray(values), meta_out)
                blocks.append(self._block_to_frame(agg, labels, extractor.feature_names()))

        if _debug_timing and _timing:
            n_cells = _timing.pop("n_cells", None)
            parts = ", ".join(f"{k} {v:.1f}s" for k, v in _timing.items())
            prefix = f"{len(batch)} fields"
            if n_cells is not None:
                prefix += f" / {int(n_cells)} cells"
            print(f"[batch-time] {prefix} | {parts}", flush=True)

        if not blocks:
            return pd.DataFrame()

        # Join the blocks on the grouping keys.
        profile = blocks[0]
        for block in blocks[1:]:
            profile = profile.merge(block, on=self.grouping_keys, how="outer")

        return profile
