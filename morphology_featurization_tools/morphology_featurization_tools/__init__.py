"""
morphology_featurization_tools: image analysis and morphological profiling.

Public names are imported lazily (PEP 562), on first access. joblib workers
import this package to unpickle their tasks; lazy imports keep them from
loading torch, cellpose and matplotlib when they only need numpy code.
"""

from __future__ import annotations

import importlib

# public name -> module that defines it
_EXPORTS: dict[str, str] = {
    # datasets (pulls in torch, jump_portrait)
    "MultiChannelImageDataset": ".datasets",
    "MicroscopyNamingConvention": ".datasets",
    "July26NamingConvention": ".datasets",
    "FlatTifNamingConvention": ".datasets",
    "FlatTifDataset": ".datasets",
    "JUMPCellPaintingDataset": ".datasets",
    "JUMPTarget2Dataset": ".datasets",
    "JUMPCompoundDataset": ".datasets",
    "JUMPPositiveControlDataset": ".datasets",
    "JUMPCRISPRDataset": ".datasets",
    # structure tensor (numpy/scipy/sklearn only)
    "StructureTensorTransformer": ".structure_tensor",
    "StructureTensorFeatureExtractor": ".structure_tensor",
    "StructureTensorSummaryStatistics": ".structure_tensor",
    "IntensityStatistics": ".structure_tensor",
    # preprocessing
    "ImagePreprocessor": ".image_preprocess_utilities",
    "DenoiseBilateralPreprocessor": ".image_preprocess_utilities",
    "FloatConverter": ".image_preprocess_utilities",
    "RescaleToZeroOne": ".image_preprocess_utilities",
    "ReduceOutlyingIntensities": ".image_preprocess_utilities",
    "ChannelSelector": ".image_preprocess_utilities",
    "IlluminationCorrection": ".illumination",
    # autoencoder (pulls in torch, openphenom, transformers)
    "featurize_dataset": ".autoencoder",
    "TypicalVariationNormalization": ".normalizers",
    "MADRobustize": ".normalizers",
    "permutation_test_mean_diff": ".utilities",
    "permutation_test_mean_diff_pipeline": ".utilities",
    "ConcatenatingTransformer": ".feature_mergers",
    # TDAExplore image featurization (pulls in tdaexplore -> gudhi)
    "TDAExploreFeaturizer": ".tdaexplore",
    "PatchFeatures": ".tdaexplore",
    "PatchAggregator": ".tdaexplore",
    # segmentation (pulls in cellpose -> torch)
    "CropSegmenter": ".segmentation",
    "FieldMaskSegmenter": ".segmentation",
    "CellposeSegmenter": ".segmentation",
    "FeatureAggregator": ".segmentation",
    # modular feature system
    "build_feature": ".features",
    "build_features": ".features",
    "available_features": ".features",
    "FeatureOrchestrator": ".features",
    "Regime": ".features",
    # diagnostics / reporting (pulls in matplotlib)
    "plate_layout_heatmap": ".diagnostics",
    "control_correlation_matrix": ".diagnostics",
    "parse_well_id": ".diagnostics",
    "get_writer": ".reporting",
    "ResultsWriter": ".reporting",
    "use_headless_backend": ".reporting",
}

# submodules reachable as attributes, e.g. morphology_featurization_tools.features
_SUBMODULES = frozenset({
    "datasets", "features", "structure_tensor",
    "image_preprocess_utilities", "illumination", "autoencoder", "utilities",
    "feature_mergers", "tdaexplore", "segmentation", "diagnostics", "reporting",
    "normalizers", "retrieval",
})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | set(_SUBMODULES))
