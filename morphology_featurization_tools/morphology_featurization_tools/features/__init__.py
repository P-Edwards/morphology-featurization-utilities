"""
Named feature extractors. build_features([...]) creates them by name and
FeatureOrchestrator runs them over batches of fields.

  per-cell (crops):        structure_tensor, intensity_stats
  per-cell (field+mask):   cp_sizeshape, cp_zernike, cp_feret, cp_intensity,
                           cp_texture, cp_radial_distribution, cp_granularity,
                           cp_colocalization, structure_tensor_field
  per-field:               autoencoder, tda

The feature modules are imported, and so registered, only when a feature is
first requested by name, so joblib workers do not import torch or cellpose.
"""

from __future__ import annotations

import importlib

from . import base as _base
from .base import (
    Regime,
    FeatureExtractor,
    register_feature,
    split_by_regime,
)

_FEATURE_MODULES = ("per_cell", "field_mask", "per_field")
_registry_populated = False


def _populate_registry() -> None:
    """Import the feature modules so their @register_feature decorators run."""
    global _registry_populated
    if _registry_populated:
        return
    for module_name in _FEATURE_MODULES:
        importlib.import_module(f".{module_name}", __name__)
    _registry_populated = True


def available_features(*args, **kwargs):
    """Names of every registered feature."""
    _populate_registry()
    return _base.available_features(*args, **kwargs)


def build_feature(*args, **kwargs):
    """Construct one feature extractor by name."""
    _populate_registry()
    return _base.build_feature(*args, **kwargs)


def build_features(*args, **kwargs):
    """Construct several feature extractors by name."""
    _populate_registry()
    return _base.build_features(*args, **kwargs)


def __getattr__(name: str):
    if name == "FeatureOrchestrator":
        from .orchestration import FeatureOrchestrator

        globals()[name] = FeatureOrchestrator
        return FeatureOrchestrator
    if name in _FEATURE_MODULES or name == "orchestration":
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"FeatureOrchestrator", "orchestration",
                                    *_FEATURE_MODULES})


__all__ = [
    "Regime",
    "FeatureExtractor",
    "register_feature",
    "available_features",
    "build_feature",
    "build_features",
    "split_by_regime",
    "FeatureOrchestrator",
]
