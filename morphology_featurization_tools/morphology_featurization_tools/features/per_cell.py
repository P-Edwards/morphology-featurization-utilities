"""
Per-cell feature extractors on stacks of single-cell crops (N, C, H, W), each
returning an (N, F) matrix: the structure tensor and intensity statistics
of this package, on one `channel` of the crops. Cells
are split into chunks across `n_jobs` worker processes.

The cp_measure CellProfiler families work on whole fields and are in
field_mask.py.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .base import FeatureExtractor, Regime, register_feature
from ..structure_tensor import (
    StructureTensorTransformer,
    StructureTensorFeatureExtractor,
    StructureTensorSummaryStatistics,
    IntensityStatistics,
)


def _select_channel(images: np.ndarray, channel: int) -> np.ndarray:
    """(N, C, H, W) -> (N, H, W) selecting one channel; pass (N, H, W) through."""
    if images.ndim == 4:
        return images[:, channel, :, :]
    return images


def _crop_to_foreground(image: np.ndarray, margin: int) -> np.ndarray:
    """
    Crop a single-cell canvas (H, W) to the bounding box of its nonzero pixels
    plus `margin`, clipped to the image. The summary statistics ignore
    background pixels, so results are unchanged when `margin` covers the
    filters' support.
    """
    foreground_rows, foreground_cols = np.nonzero(image)
    if foreground_rows.size == 0:
        return image

    height, width = image.shape
    top    = max(int(foreground_rows.min()) - margin, 0)
    bottom = min(int(foreground_rows.max()) + margin + 1, height)
    left   = max(int(foreground_cols.min()) - margin, 0)
    right  = min(int(foreground_cols.max()) + margin + 1, width)
    return image[top:bottom, left:right]


def _extract_in_parallel(
    chunk_fn,
    images: np.ndarray,
    n_jobs: int,
    cells_per_chunk: int = 256,
    pre_dispatch: str | int = "n_jobs",
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply `chunk_fn` to `images` (N, ...) across `n_jobs` worker processes and
    concatenate the (n, F) results back into one (N, F) array in input order.

    Parameters
    ----------
    chunk_fn : callable
        Maps a chunk of cells (n, ...) to a feature block (n, F). Must be
        picklable (a module-level function or a bound method).
    images : np.ndarray
        Cell stack, first axis indexed by cell.
    n_jobs : int
        Worker processes. Values <= 1 run `chunk_fn` directly.
    cells_per_chunk : int
        Cells per work unit. A fixed chunk size keeps peak worker memory near
        n_jobs x cells_per_chunk x bytes per cell, whatever the cell count.
    pre_dispatch : str or int
        Passed to joblib; how many chunks are prepared ahead of the workers.
    verbose : bool
        Print the chunking decision for this call.
    """
    n_images = len(images)

    if n_images == 0:
        return chunk_fn(images)

    if n_jobs is None or n_jobs <= 1 or n_images < 2 * n_jobs:
        if verbose:
            print(f"[parallel] SERIAL: {n_images} cells, n_jobs={n_jobs}", flush=True)
        return chunk_fn(images)

    # At least one chunk per worker.
    n_chunks = max(int(np.ceil(n_images / cells_per_chunk)), min(n_jobs, n_images))
    chunks = np.array_split(images, n_chunks)

    if verbose:
        print(f"[parallel] PARALLEL: {n_images} cells -> {n_chunks} chunks "
              f"of ~{int(np.ceil(n_images / n_chunks))} across {n_jobs} workers",
              flush=True)

    results = Parallel(n_jobs=n_jobs, prefer="processes", pre_dispatch=pre_dispatch)(
        delayed(chunk_fn)(chunk) for chunk in chunks
    )

    return np.concatenate([np.asarray(r) for r in results], axis=0)


class _ChunkedExtraction:
    """Chunking settings shared by the per-cell extractors (see
    _extract_in_parallel)."""

    def _configure_chunking(
        self,
        cells_per_chunk: int = 256,
        pre_dispatch: str | int = "n_jobs",
        verbose: bool = False,
    ) -> None:
        self.cells_per_chunk = cells_per_chunk
        self.pre_dispatch = pre_dispatch
        self.verbose = verbose

    def _run_chunked(self, chunk_fn, images: np.ndarray, n_jobs: int) -> np.ndarray:
        return _extract_in_parallel(
            chunk_fn,
            images,
            n_jobs,
            cells_per_chunk=getattr(self, "cells_per_chunk", 256),
            pre_dispatch=getattr(self, "pre_dispatch", "n_jobs"),
            verbose=getattr(self, "verbose", False),
        )


@register_feature("structure_tensor")
class StructureTensorFeature(_ChunkedExtraction, FeatureExtractor):
    """
    Structure-tensor orientation/coherence summary statistics (this package's
    StructureTensor* chain) computed per cell on a chosen channel.
    """
    regime = Regime.PER_CELL

    def __init__(
        self,
        channel: int = 0,
        sigma_gradient: float = 1.0,
        sigma_integration: float = 2.0,
        cells_per_chunk: int = 256,
        pre_dispatch: str | int = "n_jobs",
        verbose: bool = False,
    ):
        self._configure_chunking(cells_per_chunk, pre_dispatch, verbose)
        self.channel = channel
        self._transform = StructureTensorTransformer(
            sigma_gradient=sigma_gradient, sigma_integration=sigma_integration
        )
        self._features = StructureTensorFeatureExtractor()
        self._summary = StructureTensorSummaryStatistics(as_pandas_dataframe=False)

        # A pixel's structure tensor depends on values up to
        # 3 * (sigma_gradient + sigma_integration) away.
        self._crop_margin = int(np.ceil(3 * (sigma_gradient + sigma_integration)))

    def feature_names(self) -> list[str]:
        return [f"structure_tensor__{c}" for c in self._summary.column_names_]

    def _extract_chunk(self, single: np.ndarray) -> np.ndarray:
        """Structure-tensor summary of each cell in a chunk, each cropped to
        its own bounding box first."""
        rows = []
        for index in range(len(single)):
            cell = _crop_to_foreground(single[index], self._crop_margin)
            tensor = self._transform.transform(cell[np.newaxis])
            feats = self._features.transform(tensor)
            rows.append(np.asarray(self._summary.transform(feats))[0])

        if not rows:
            return np.zeros((0, len(self._summary.column_names_)))
        return np.stack(rows)

    def extract(self, images: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        single = _select_channel(images, self.channel)
        return self._run_chunked(self._extract_chunk, single, n_jobs)


@register_feature("intensity_stats")
class IntensityStatsFeature(_ChunkedExtraction, FeatureExtractor):
    """
    Basic per-cell intensity distribution statistics (this package's
    IntensityStatistics) on a chosen channel.
    """
    regime = Regime.PER_CELL

    def __init__(
        self,
        channel: int = 0,
        cells_per_chunk: int = 256,
        pre_dispatch: str | int = "n_jobs",
        verbose: bool = False,
    ):
        self._configure_chunking(cells_per_chunk, pre_dispatch, verbose)
        self.channel = channel
        self._stats = IntensityStatistics(as_pandas_dataframe=False)

    def feature_names(self) -> list[str]:
        names = getattr(self._stats, "FEATURE_NAMES", None)
        if names is None:
            names = [f"stat{i}" for i in range(6)]
        return [f"intensity__{n}" for n in names]

    def _extract_chunk(self, single: np.ndarray) -> np.ndarray:
        return np.asarray(self._stats.transform(single))

    def extract(self, images: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        single = _select_channel(images, self.channel)
        return self._run_chunked(self._extract_chunk, single, n_jobs)
