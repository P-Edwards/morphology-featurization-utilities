"""
Per-cell features measured on whole fields with their integer label masks
(FieldMaskSegmenter output): the cp_measure CellProfiler families, and the
structure tensor.

cp_measure measures every object of a field in one call: each function takes
(label_mask, intensity_image) and returns feature name -> one value per label.
extract_fields returns per-cell rows and the (field_index, label) of each row.

cp_measure is optional and imported lazily (pip install cp-measure). Its core
measurement groups are radial_distribution, radial_zernikes, intensity,
sizeshape, zernike, feret, texture and granularity; two-channel correlations
come from get_correlation_measurements().
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import pandas as pd

from joblib import Parallel, delayed

from .base import Regime, register_feature


def _load_core_measurements() -> dict:
    try:
        from cp_measure.bulk import get_core_measurements
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cp_measure is required for CellProfiler-style features. "
            "Install it with `pip install cp-measure`."
        ) from exc
    return get_core_measurements()


def _load_correlation_measurements() -> dict:
    try:
        from cp_measure.bulk import get_correlation_measurements
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cp_measure is required for CellProfiler-style features. "
            "Install it with `pip install cp-measure`."
        ) from exc
    return get_correlation_measurements()


def _normalize_to_unit_float(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] by the image maximum, as cp_measure expects floats in
    [0, 1]."""
    image = image.astype(np.float64)
    peak = image.max()
    return image / peak if peak > 0 else image


class FieldMaskFeatureExtractor:
    """
    Base class for cp_measure-backed feature extractors operating on
    (field, mask) pairs and producing one row per labelled cell.

    Subclasses either select a group of single-channel measurements
    (SingleChannelCPFeature) or a two-channel correlation measurement
    (CorrelationCPFeature).
    """
    regime = Regime.PER_CELL

    name: str = ""

    def feature_names(self) -> list[str]:
        raise NotImplementedError

    def extract_fields(
        self,
        fields: list[np.ndarray],
        masks: list[np.ndarray],
        n_jobs: int = 1,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Measure all cells across all fields.

        Parameters
        ----------
        fields : list of np.ndarray
            Field images, (H, W) or (C, H, W).
        masks : list of np.ndarray
            Integer label masks (H, W), contiguous labels, aligned with
            fields.
        n_jobs : int
            Parallel workers across fields.

        Returns
        -------
        features : np.ndarray, shape (total_cells, F)
        provenance : list of (field_index, label)
            The originating field index and object label for each row, so the
            caller can attach that field's metadata to each cell.
        """
        raise NotImplementedError


class SingleChannelCPFeature(FieldMaskFeatureExtractor):
    """
    A group of cp_measure single-channel measurements (e.g. intensity,
    texture, sizeshape) computed on one channel of each field.

    Parameters
    ----------
    measurement_keys : list of str
        Names of cp_measure core measurement groups to include.
    channel : int
        Channel index to measure (for multichannel fields). Ignored for
        single-channel fields.
    short : str
        Column-name prefix.
    """
    regime = Regime.PER_CELL

    def __init__(self, measurement_keys: list[str], channel: int, short: str):
        self.measurement_keys = measurement_keys
        self.channel = channel
        self.short = short
        self._functions = None
        self._names: list[str] | None = None

    def _get_functions(self) -> dict:
        if self._functions is None:
            core = _load_core_measurements()
            missing = [k for k in self.measurement_keys if k not in core]
            if missing:
                raise KeyError(
                    f"cp_measure has no measurement(s) {missing}. "
                    f"Available: {sorted(core)}"
                )
            self._functions = {k: core[k] for k in self.measurement_keys}
        return self._functions

    def _field_channel(self, field: np.ndarray) -> np.ndarray:
        if field.ndim == 3:
            return field[self.channel]
        return field

    def _measure_field(
        self, field: np.ndarray, mask: np.ndarray
    ) -> tuple[dict, int]:
        """Measure the configured channel of one (C, H, W) or (H, W) field."""
        return self._measure_image(self._field_channel(field), mask)

    def _measure_image(
        self, image_2d: np.ndarray, mask: np.ndarray
    ) -> tuple[dict, int]:
        """Measure one 2D image; return (name -> array, n_labels)."""
        functions = self._get_functions()
        n_labels = int(mask.max())
        if n_labels == 0:
            return {}, 0

        image = _normalize_to_unit_float(image_2d)
        combined: dict = {}
        for key, fn in functions.items():
            try:
                result = fn(mask, image)
            except Exception:
                result = {}
            for feat_name, values in result.items():
                combined[f"{self.short}__{key}__{feat_name}"] = np.asarray(values)
        return combined, n_labels

    def feature_names(self) -> list[str]:
        return list(self._names) if self._names is not None else []

    def extract_fields(
        self,
        fields: list[np.ndarray],
        masks: list[np.ndarray],
        n_jobs: int = 1,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Only the measured channel is sent to the workers.
        per_field = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(self._measure_image)(image, masks[i])
            for i, image in enumerate(self.prepare(fields))
        )
        return self.assemble(per_field)

    def prepare(self, fields):
        """One measured 2D plane per field, ready to dispatch."""
        return [self._field_channel(field) for field in fields]

    def measure_prepared(self, prepared, mask):
        return self._measure_image(prepared, mask)

    def assemble(self, per_field):
        """Turn per-field (dict, n_labels) results into (values, provenance).
        Separate from extract_fields so a caller (the orchestrator) can
        schedule the per-field measurements itself."""
        names: list[str] | None = None
        for combined, n_labels in per_field:
            if combined:
                names = sorted(combined.keys())
                break
        if names is None:
            self._names = []
            return np.zeros((0, 0)), []
        self._names = names

        rows: list[np.ndarray] = []
        provenance: list[tuple[int, int]] = []
        for field_index, (combined, n_labels) in enumerate(per_field):
            if n_labels == 0:
                continue
            for label_position in range(n_labels):
                row = np.array([
                    float(combined[name][label_position])
                    if name in combined and label_position < len(combined[name])
                    else 0.0
                    for name in names
                ])
                rows.append(row)
                provenance.append((field_index, label_position + 1))

        features = np.vstack(rows) if rows else np.zeros((0, len(names)))
        return features, provenance


class CorrelationCPFeature(FieldMaskFeatureExtractor):
    """
    cp_measure two-channel correlation / colocalization measurements between
    a pair of channels of each field.

    Parameters
    ----------
    channel_a, channel_b : int
        Channel indices for the correlation pair.
    """
    regime = Regime.PER_CELL

    def __init__(self, channel_a: int = 0, channel_b: int = 1,
                 channel_a_name: str | None = None,
                 channel_b_name: str | None = None):
        self.channel_a = channel_a
        self.channel_b = channel_b
        # Named after both channels, as in CellProfiler
        # (Cells_Correlation_Correlation_DNA_AGP).
        pair = (f"{channel_a_name}_{channel_b_name}"
                if channel_a_name and channel_b_name else None)
        self.short = "cp_coloc" if pair is None else f"cp_coloc_{pair}"
        self._functions = None
        self._names: list[str] | None = None

    def _get_functions(self) -> dict:
        if self._functions is None:
            self._functions = _load_correlation_measurements()
        return self._functions

    def _measure_field(
        self, field: np.ndarray, mask: np.ndarray
    ) -> tuple[dict, int]:
        if field.ndim != 3:
            raise ValueError("cp_colocalization requires multichannel fields.")
        n_labels = int(mask.max())
        if n_labels == 0:
            return {}, 0

        return self._measure_pair(field[self.channel_a], field[self.channel_b], mask)

    def _measure_pair(
        self, plane_a: np.ndarray, plane_b: np.ndarray, mask: np.ndarray
    ) -> tuple[dict, int]:
        """Measure one pair of 2D channel images."""
        n_labels = int(mask.max())
        if n_labels == 0:
            return {}, 0

        image_a = _normalize_to_unit_float(plane_a)
        image_b = _normalize_to_unit_float(plane_b)

        combined: dict = {}
        for key, fn in self._get_functions().items():
            try:
                # The correlation functions take (pixels_1, pixels_2, masks),
                # mask last; the single-channel ones take (masks, pixels).
                result = fn(image_a, image_b, mask)
            except Exception:
                result = {}
            for feat_name, values in result.items():
                combined[f"{self.short}__{key}__{feat_name}"] = np.asarray(values)
        return combined, n_labels

    def feature_names(self) -> list[str]:
        return list(self._names) if self._names is not None else []

    def extract_fields(
        self,
        fields: list[np.ndarray],
        masks: list[np.ndarray],
        n_jobs: int = 1,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Only the two measured channels are sent to the workers.
        per_field = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(self._measure_pair)(pair[0], pair[1], masks[i])
            for i, pair in enumerate(self.prepare(fields))
        )
        return self.assemble(per_field)

    def prepare(self, fields):
        """The two measured planes of each field."""
        return [(field[self.channel_a], field[self.channel_b]) for field in fields]

    def measure_prepared(self, prepared, mask):
        return self._measure_pair(prepared[0], prepared[1], mask)

    def assemble(self, per_field):
        """Turn per-field (dict, n_labels) results into (values, provenance)."""
        names: list[str] | None = None
        for combined, n_labels in per_field:
            if combined:
                names = sorted(combined.keys())
                break
        if names is None:
            self._names = []
            return np.zeros((0, 0)), []
        self._names = names

        rows: list[np.ndarray] = []
        provenance: list[tuple[int, int]] = []
        for field_index, (combined, n_labels) in enumerate(per_field):
            if n_labels == 0:
                continue
            for label_position in range(n_labels):
                row = np.array([
                    float(combined[name][label_position])
                    if name in combined and label_position < len(combined[name])
                    else 0.0
                    for name in names
                ])
                rows.append(row)
                provenance.append((field_index, label_position + 1))

        features = np.vstack(rows) if rows else np.zeros((0, len(names)))
        return features, provenance




class PerCellInFieldFeature(FieldMaskFeatureExtractor):
    """
    Per-cell features computed from (field, mask) without fixed-size crops.

    Each cell is cut out as its bounding box plus `margin` pixels, with every
    pixel outside the cell set to zero, as the crop segmenter does. These
    features skip the crop stage and are scheduled with the cp_measure
    families. Near a field edge the margin is clipped to the field rather than
    to a crop canvas, so such cells can differ slightly from the crop path.

    Subclasses implement `measure_cell`.
    """

    #: Extra pixels kept around a cell's bounding box, covering the support of
    #: whatever filters `measure_cell` applies.
    margin: int = 0
    #: Prefix for feature columns.
    short: str = ""

    def __init__(self, channel: int = 0, channel_name: str | None = None):
        self.channel = channel
        self.channel_name = channel_name
        self._names: list[str] | None = None

    # -- to implement ------------------------------------------------------

    def measure_cell(self, cell: np.ndarray) -> dict:
        """Measure one masked, cropped cell; return name -> scalar."""
        raise NotImplementedError

    # -- scheduling interface (mirrors SingleChannelCPFeature) --------------

    def _field_channel(self, field: np.ndarray) -> np.ndarray:
        return field if field.ndim == 2 else field[self.channel]

    def prepare(self, fields):
        return [self._field_channel(field) for field in fields]

    def measure_prepared(self, prepared, mask):
        return self._measure_image(prepared, mask)

    def _measure_image(self, image_2d: np.ndarray, mask: np.ndarray):
        n_labels = int(mask.max())
        if n_labels == 0:
            return {}, 0

        # Every label's bounding box in one pass over the mask.
        boxes = ndimage.find_objects(mask)
        height, width = mask.shape
        per_label: list[dict] = []
        for label in range(1, n_labels + 1):
            box = boxes[label - 1] if label - 1 < len(boxes) else None
            if box is None:
                per_label.append({})
                continue
            rows, cols = box
            top = max(rows.start - self.margin, 0)
            bottom = min(rows.stop + self.margin, height)
            left = max(cols.start - self.margin, 0)
            right = min(cols.stop + self.margin, width)

            window = image_2d[top:bottom, left:right]
            # Zero everything that is not this cell.
            cell = np.where(mask[top:bottom, left:right] == label, window, 0)
            try:
                per_label.append(self.measure_cell(cell))
            except Exception:
                per_label.append({})

        names: list[str] = []
        for entry in per_label:
            if entry:
                names = sorted(entry.keys())
                break
        if not names:
            return {}, n_labels

        combined = {
            f"{self.short}__{name}": np.array(
                [float(entry.get(name, 0.0)) for entry in per_label]
            )
            for name in names
        }
        return combined, n_labels

    def feature_names(self) -> list[str]:
        return list(self._names) if self._names is not None else []

    def extract_fields(self, fields, masks, n_jobs: int = 1):
        per_field = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(self._measure_image)(image, masks[i])
            for i, image in enumerate(self.prepare(fields))
        )
        return self.assemble(per_field)

    assemble = SingleChannelCPFeature.assemble


@register_feature("structure_tensor_field")
class StructureTensorFieldFeature(PerCellInFieldFeature):
    """
    Structure-tensor summary statistics per cell, from (field, mask).

    Gives the same values as the crop-based `structure_tensor` feature (same
    transformer chain on the same masked window), but its columns are sorted
    by name rather than in StructureTensorSummaryStatistics order.
    """

    short = "structure_tensor"

    def __init__(
        self,
        channel: int = 0,
        channel_name: str | None = None,
        sigma_gradient: float = 1.0,
        sigma_integration: float = 2.0,
    ):
        super().__init__(channel=channel, channel_name=channel_name)
        from ..structure_tensor import (
            StructureTensorTransformer,
            StructureTensorFeatureExtractor,
            StructureTensorSummaryStatistics,
        )
        self.sigma_gradient = sigma_gradient
        self.sigma_integration = sigma_integration
        self._transform = StructureTensorTransformer(
            sigma_gradient=sigma_gradient, sigma_integration=sigma_integration
        )
        self._features = StructureTensorFeatureExtractor()
        self._summary = StructureTensorSummaryStatistics(as_pandas_dataframe=False)
        # A pixel's structure tensor depends on values up to
        # 3 * (sigma_gradient + sigma_integration) away.
        self.margin = int(np.ceil(3 * (sigma_gradient + sigma_integration)))

    def measure_cell(self, cell: np.ndarray) -> dict:
        tensor = self._transform.transform(cell[np.newaxis])
        feats = self._features.transform(tensor)
        row = np.asarray(self._summary.transform(feats))[0]
        return dict(zip(self._summary.column_names_, row))


def _short_with_channel(base: str, channel_name: str | None) -> str:
    """Feature-name prefix including the channel, as CellProfiler names
    channel-dependent measurements (Cells_Intensity_MeanIntensity_AGP);
    mask-derived families pass no channel name."""
    return base if channel_name is None else f"{base}_{channel_name}"


# ----------------------------------------------------------------------

@register_feature("cp_sizeshape")
class CPSizeShape(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["sizeshape"], channel,
                         _short_with_channel("cp_sizeshape", channel_name))


@register_feature("cp_zernike")
class CPZernike(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["zernike"], channel,
                         _short_with_channel("cp_zernike", channel_name))


@register_feature("cp_feret")
class CPFeret(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["feret"], channel,
                         _short_with_channel("cp_feret", channel_name))


@register_feature("cp_intensity")
class CPIntensity(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["intensity"], channel,
                         _short_with_channel("cp_intensity", channel_name))


@register_feature("cp_texture")
class CPTexture(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["texture"], channel,
                         _short_with_channel("cp_texture", channel_name))


@register_feature("cp_radial_distribution")
class CPRadialDistribution(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(
            ["radial_distribution", "radial_zernikes"], channel,
            _short_with_channel("cp_radial", channel_name),
        )


@register_feature("cp_granularity")
class CPGranularity(SingleChannelCPFeature):
    def __init__(self, channel: int = 0, channel_name: str | None = None):
        super().__init__(["granularity"], channel,
                         _short_with_channel("cp_granularity", channel_name))


@register_feature("cp_colocalization")
class CPColocalization(CorrelationCPFeature):
    pass
