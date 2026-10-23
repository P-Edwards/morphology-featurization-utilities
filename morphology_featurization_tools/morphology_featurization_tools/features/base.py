"""
Feature registry. Each feature type implements FeatureExtractor and is
registered under a name with @register_feature; build_features([...]) creates
extractors by name.

  PER_CELL   computed on segmented cells, aggregated over cells to the well
  PER_FIELD  computed on whole fields, aggregated over fields to the well
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class Regime(Enum):
    """Operating regime of a feature extractor."""
    PER_CELL = "per_cell"
    PER_FIELD = "per_field"


@runtime_checkable
class FeatureExtractor(Protocol):
    """
    Common interface for all feature types.

    An extractor turns a stack of images into a numeric feature matrix plus
    the names of its columns. Whether the images are cell crops or whole
    fields is determined by the extractor's `regime`; the orchestration
    supplies the appropriate stack.

    Implementations should be cheap to construct and hold their
    configuration as attributes, so they can be listed, named, and combined
    without doing any work until `extract` is called.
    """

    name: str
    regime: Regime

    def feature_names(self) -> list[str]:
        """Ordered names of the columns `extract` will produce."""
        ...

    def extract(self, images: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """
        Compute features for a stack of images.

        Parameters
        ----------
        images : np.ndarray
            (N, H, W) or (N, C, H, W) stack. For PER_CELL extractors these
            are cell crops; for PER_FIELD they are whole fields.
        n_jobs : int
            Number of parallel workers to use where the extractor supports
            parallelism. 1 means serial.

        Returns
        -------
        np.ndarray of shape (N, len(self.feature_names())).
        """
        ...


# Feature name -> factory; factories take keyword overrides.
_REGISTRY: dict[str, Callable[..., FeatureExtractor]] = {}


def register_feature(name: str) -> Callable:
    """
    Class decorator registering a FeatureExtractor factory under `name`.

    The decorated class is stored as its own factory; build_features
    instantiates it with any per-name overrides supplied by the caller.
    """
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Feature '{name}' is already registered.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def available_features() -> list[str]:
    """Names of all registered feature types."""
    return sorted(_REGISTRY)


def build_feature(name: str, **overrides) -> FeatureExtractor:
    """Instantiate a single registered feature extractor by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown feature '{name}'. Available: {available_features()}"
        )
    return _REGISTRY[name](**overrides)


def build_features(
    names: list[str],
    overrides: dict[str, dict] | None = None,
) -> list[FeatureExtractor]:
    """
    Instantiate several registered feature extractors by name.

    Parameters
    ----------
    names : list of str
        Feature names to build, e.g. ["structure_tensor", "cp_intensity"].
    overrides : dict, optional
        Mapping of feature name -> kwargs dict passed to that feature's
        constructor, letting you tune individual features without changing
        the others.

    Returns
    -------
    list of FeatureExtractor, in the order requested.
    """
    overrides = overrides or {}
    return [build_feature(n, **overrides.get(n, {})) for n in names]


def split_by_regime(
    extractors: list[FeatureExtractor],
) -> tuple[list[FeatureExtractor], list[FeatureExtractor]]:
    """Partition extractors into (per_cell, per_field) lists, order preserved."""
    per_cell = [e for e in extractors if e.regime is Regime.PER_CELL]
    per_field = [e for e in extractors if e.regime is Regime.PER_FIELD]
    return per_cell, per_field
