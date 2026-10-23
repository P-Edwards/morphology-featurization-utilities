"""
Persistence-landscape features of image datasets, computed with the Python
tdaexplore package.

  TDAExploreFeaturizer.featurize(dataset)          one mean landscape vector
                                                   per image
  TDAExploreFeaturizer.featurize_patches(dataset)  one landscape vector per
                                                   patch, with the index of its
                                                   image (PatchFeatures)
  PatchAggregator                                  summarizes patch-level values
                                                   per image, for CV loops that
                                                   split on images

Multi-channel datasets: set channel=<int>; run one featurizer per channel and
concatenate the image-level outputs (ConcatenatingTransformer).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# PatchFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatchFeatures:
    """
    Patch-level landscape features for an entire dataset.

    Attributes
    ----------
    X : np.ndarray, shape (total_patches, n_features)
        Raw landscape feature vectors, one row per patch.
    image_idx : np.ndarray, shape (total_patches,), dtype int
        Zero-based index into `metadata` identifying the source image
        for each patch.  Stable across channels and CV folds.
    metadata : pd.DataFrame, shape (n_images, 2)
        Per-image metadata with columns ["class", "batch"], indexed
        0 .. n_images-1 to match image_idx values.
    """
    X:          np.ndarray
    image_idx:  np.ndarray
    metadata:   pd.DataFrame


# ---------------------------------------------------------------------------
# PatchAggregator
# ---------------------------------------------------------------------------

class PatchAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate patch-level landscape vectors to image-level feature vectors.

    For each image (identified by a contiguous block of image_idx values)
    the following statistics are computed per feature dimension:
        mean, Q1 (25th percentile), median (50th), Q3 (75th percentile)

    Output width = n_patch_features * 4.

    Designed for use inside a CV loop where patches are pre-split by image:

        aggregator = PatchAggregator()
        X_image = aggregator.transform(patch_X, image_idx=patch_image_idx)

    Args
    ----
    stats : list[str] | None
        Subset of ["mean", "q1", "median", "q3"] to compute.
        Default is all four.
    """

    STAT_FUNCTIONS = {
        "mean":   lambda a: np.mean(a, axis=0),
        "q1":     lambda a: np.percentile(a, 25, axis=0),
        "median": lambda a: np.median(a, axis=0),
        "q3":     lambda a: np.percentile(a, 75, axis=0),
    }
    STAT_ORDER = ["mean", "q1", "median", "q3"]

    def __init__(self, stats: list[str] | None = None):
        self.stats = stats if stats is not None else self.STAT_ORDER

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(
        self,
        X: np.ndarray,
        image_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (total_patches, n_features)
        image_idx : np.ndarray, shape (total_patches,), dtype int
            Must be provided.

        Returns
        -------
        np.ndarray, shape (n_images, n_features * n_stats)
            Rows are in ascending image_idx order.
        """
        if image_idx is None:
            raise ValueError("image_idx must be provided to PatchAggregator.transform().")

        invalid = [s for s in self.stats if s not in self.STAT_FUNCTIONS]
        if invalid:
            raise ValueError(f"Unknown stats: {invalid}. "
                             f"Valid options: {list(self.STAT_FUNCTIONS)}")

        unique_images = np.unique(image_idx)
        stat_fns      = [self.STAT_FUNCTIONS[s] for s in self.stats]

        rows = []
        for img in unique_images:
            patches = X[image_idx == img]          # (n_patches_i, n_features)
            row = np.concatenate([fn(patches) for fn in stat_fns])
            rows.append(row)

        return np.stack(rows).astype(np.float32)

    def predict(self, X, **kwargs):
        return self.transform(X, **kwargs)

    @staticmethod
    def feature_names(n_patch_features: int, stats: list[str] | None = None) -> list[str]:
        """Return column names matching the transform output."""
        if stats is None:
            stats = PatchAggregator.STAT_ORDER
        return [
            f"feat_{i}_{stat}"
            for stat in stats
            for i in range(n_patch_features)
        ]


# ---------------------------------------------------------------------------
# TDAExploreFeaturizer
# ---------------------------------------------------------------------------

class TDAExploreFeaturizer:
    """
    Featurize a dataset with tdaexplore persistence landscapes.

    Patches of radius `radius` are sampled from each image, an alpha-complex
    persistence landscape is computed for each, and the landscapes are either
    averaged per image (featurize) or returned per patch (featurize_patches).
    A landscape vector has 2 x number_of_layers x 401 entries.

    Args
    ----
    radius : int
        Patch radius in pixels (default 50).
    n_jobs : int
        Images featurized in parallel (default 1).
    channel : int | None
        Channel to featurize from (C, H, W) images; None for datasets whose
        images are already (H, W).
    cache_path : str | Path | None
        If the file exists, features are loaded from it instead of computed;
        otherwise they are computed and saved there. A cache written by one
        mode cannot be read by the other.
    number_of_patches : int | None
        Patches per image. None derives it from each image's area:
        patch_ratio x area / (pi x radius^2), at least 2.
    patch_ratio, number_of_layers, proportion_of_patch_sparse,
    noise_threshold :
        Passed to tdaexplore.PatchLandscapeTransformer; the defaults are
        tdaexplore's.
    random_state : int | None
        Seed for patch sampling; each image gets its own stream derived from
        it, so results do not depend on n_jobs.
    """

    def __init__(
        self,
        radius: int = 50,
        n_jobs: int = 1,
        channel: int | None = None,
        cache_path: str | Path | None = None,
        number_of_patches: int | None = None,
        patch_ratio: float = 2.0,
        number_of_layers: int = 50,
        proportion_of_patch_sparse: float = 0.025,
        noise_threshold: float = 0.05,
        random_state: int | None = None,
    ):
        self.radius = radius
        self.n_jobs = n_jobs
        self.channel = channel
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.number_of_patches = number_of_patches
        self.patch_ratio = patch_ratio
        self.number_of_layers = number_of_layers
        self.proportion_of_patch_sparse = proportion_of_patch_sparse
        self.noise_threshold = noise_threshold
        self.random_state = random_state

    def featurize(self, dataset) -> tuple[np.ndarray, pd.DataFrame]:
        """
        One averaged landscape vector per image.

        Returns
        -------
        X        : np.ndarray, shape (n_images, n_features), float32
        metadata : pd.DataFrame, columns ["class", "batch"]
        """
        if self.cache_path is not None and self.cache_path.exists():
            return self._load_mean(self.cache_path)

        blocks, metadata = self._landscapes(dataset)
        X = np.vstack([np.asarray(block.mean(axis=0)) for block in blocks])
        X = X.astype(np.float32)

        if self.cache_path is not None:
            self._save_mean(X, metadata, self.cache_path)
        return X, metadata

    def featurize_patches(self, dataset) -> PatchFeatures:
        """
        One landscape vector per patch.

        Returns
        -------
        PatchFeatures with
            X          : (total_patches, n_features), float32
            image_idx  : (total_patches,) int, 0-based index into metadata
            metadata   : (n_images,) DataFrame with columns ["class", "batch"]
        """
        if self.cache_path is not None and self.cache_path.exists():
            return self._load_patches(self.cache_path)

        blocks, metadata = self._landscapes(dataset)
        X = np.vstack([block.toarray().astype(np.float32) for block in blocks])
        image_idx = np.concatenate([
            np.full(block.shape[0], i, dtype=np.int64) for i, block in enumerate(blocks)
        ])
        patch_features = PatchFeatures(X=X, image_idx=image_idx, metadata=metadata)

        if self.cache_path is not None:
            self._save_patches(patch_features, self.cache_path)
        return patch_features

    def _image(self, item) -> np.ndarray:
        image = np.asarray(item["image"])
        if image.ndim == 3:
            if self.channel is None:
                raise ValueError(
                    "Images have a channel axis; set channel to the one to "
                    "featurize (one TDAExploreFeaturizer per channel).")
            image = image[self.channel]
        return image.astype(float)

    def _landscapes(self, dataset):
        """(one sparse patch-landscape matrix per image, metadata)."""
        from joblib import Parallel, delayed
        from sklearn.base import clone
        from tdaexplore import PatchLandscapeTransformer

        transformer = PatchLandscapeTransformer(
            number_of_patches=self.number_of_patches,
            pixel_radius_for_patches=self.radius,
            proportion_of_patch_sparse=self.proportion_of_patch_sparse,
            noise_threshold=self.noise_threshold,
            number_of_layers=self.number_of_layers,
            patch_ratio=self.patch_ratio,
            n_jobs=1,
        )

        items = [dataset[i] for i in range(len(dataset))]
        metadata = pd.DataFrame(
            {"class": [item["class"] for item in items],
             "batch": [item["batch"] for item in items]})
        images = [self._image(item) for item in items]
        del items

        if self.random_state is None:
            seeds = [None] * len(images)
        else:
            seeds = np.random.default_rng(self.random_state).spawn(len(images))

        def one_image(image, seed):
            return clone(transformer).set_params(random_state=seed).transform([image])

        blocks = Parallel(n_jobs=self.n_jobs)(
            delayed(one_image)(image, seed) for image, seed in zip(images, seeds))
        return blocks, metadata

    @staticmethod
    def _load_mean(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
        df = pd.read_csv(path, dtype={"class": str, "batch": str})
        feature_cols = [c for c in df.columns if c not in ("class", "batch")]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        metadata = df[["class", "batch"]].reset_index(drop=True)
        return X, metadata

    @staticmethod
    def _save_mean(X: np.ndarray, metadata: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
        df = pd.concat(
            [metadata.reset_index(drop=True),
             pd.DataFrame(X, columns=feature_cols)],
            axis=1,
        )
        df.to_csv(path, index=False)

    @staticmethod
    def _load_patches(path: str | Path) -> PatchFeatures:
        df = pd.read_csv(path, dtype={"class": str, "batch": str})
        if "image_idx" not in df.columns:
            raise KeyError(
                f"Cache file '{path}' has no 'image_idx' column, so it holds "
                "image means, not patches. Delete it or use another cache path.")
        feature_cols = [c for c in df.columns if c not in ("image_idx", "class", "batch")]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        image_idx = df["image_idx"].to_numpy(dtype=np.int64)
        meta = (
            df[["image_idx", "class", "batch"]]
            .drop_duplicates("image_idx")
            .sort_values("image_idx")
            .set_index("image_idx")[["class", "batch"]]
        )
        return PatchFeatures(X=X, image_idx=image_idx, metadata=meta)

    @staticmethod
    def _save_patches(pf: PatchFeatures, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_cols = [f"feat_{i}" for i in range(pf.X.shape[1])]
        meta_expanded = pf.metadata.iloc[pf.image_idx].reset_index(drop=True)
        df = pd.concat(
            [pd.Series(pf.image_idx, name="image_idx"),
             meta_expanded,
             pd.DataFrame(pf.X, columns=feature_cols)],
            axis=1,
        )
        df.to_csv(path, index=False)
