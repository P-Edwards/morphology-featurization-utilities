# Modular feature-extraction system

This package provides a named, modular feature-extraction system for
morphological high-content screening. Request feature types by name, run them
across a batch of fields, and get a concatenated well-level profile suitable
for batch correction (TVN) and copairs mAP retrieval.

## JUMP Cell Painting datasets

`JUMPCellPaintingDataset(site_table)` streams the sites of a site table from
S3; each item is one field as a (C, H, W) array. Channels default to DNA, ER,
RNA, AGP, Mito, in that order. Ready-made site tables:

```python
from morphology_featurization_tools.datasets import (
    JUMPTarget2Dataset, JUMPCompoundDataset, JUMPPositiveControlDataset,
    JUMPCRISPRDataset)

JUMPTarget2Dataset()                          # every Target-2 well, no source_9
JUMPCompoundDataset(["JCP2022_012818",        # any identifier broad_babel
                     "SRVFFFJZQVENJC-UHFFFAOYSA-N"])   # translates
JUMPPositiveControlDataset(controls="batches")
JUMPCRISPRDataset(["PLK1", "7157"])           # gene symbols, NCBI gene ids
```

`exclude` takes source names or (source, plate type) pairs; `controls`
chooses the negative controls added: those of the plates carrying the
perturbations (default), of every plate in their batches, of every selected
plate, or none. `site_table_for_wells(wells, channels)` in
`morphology_featurization_tools.datasets.jump_cell_painting` builds a site
table for any set of wells.

## Requesting features by name

```python
from morphology_featurization_tools import available_features, build_features

available_features()
# ['autoencoder', 'cp_colocalization', 'cp_feret', 'cp_granularity',
#  'cp_intensity', 'cp_radial_distribution', 'cp_sizeshape',
#  'cp_texture', 'cp_zernike', 'intensity_stats', 'structure_tensor',
#  'structure_tensor_field', 'tda']

extractors = build_features(
    ["structure_tensor", "intensity_stats", "cp_intensity", "cp_texture"],
    overrides={"structure_tensor": {"channel": 1}, "cp_intensity": {"channel": 1}},
)
```

## Regimes and input forms

Every feature declares a **regime** that determines how it is run and where it
is aggregated:

| Regime      | Input form                | Examples                                   | Segmenter            |
|-------------|---------------------------|--------------------------------------------|----------------------|
| PER_CELL    | fixed-canvas cell crops   | `structure_tensor`, `intensity_stats`      | `CropSegmenter`      |
| PER_CELL    | field + label mask        | `cp_*` (cp_measure)                        | `FieldMaskSegmenter` |
| PER_FIELD   | whole field of view       | `autoencoder`, `tda`                       | none                 |

Per-cell features are aggregated across cells to the grouping unit (well).
Per-field features are aggregated across each unit's fields.

The two per-cell input forms exist because cp_measure (CellProfiler-style
features) is designed to measure every object in a field in one call using the
integer label mask, which is far more efficient than per-crop calls and more
faithful for spatially-defined features (radial distribution, texture,
granularity). The structure-tensor and intensity featurizers use fixed-canvas
crops.

## Segmenter variants

- `CropSegmenter.segment(batch)` → `(cell_crops, cell_metadata)`; fixed-size
  canvases, mask-centroid-centered, background zeroed. (`CellposeSegmenter` is
  an alias of `CropSegmenter`.)
- `FieldMaskSegmenter.segment_fields(batch)` → `(fields, label_masks,
  field_metadata)`; whole fields with contiguous-label masks for cp_measure.
  It inherits the crop machinery, so one configured segmenter can produce both.

## Orchestration

```python
from morphology_featurization_tools import (
    CropSegmenter, FeatureOrchestrator, FieldMaskSegmenter)

orch = FeatureOrchestrator(
    extractors=extractors,
    grouping_keys=["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
    crop_segmenter=CropSegmenter(canvas_size=256, cellpose_channels=[0, 1, None], gpu=True),
    field_mask_segmenter=FieldMaskSegmenter(canvas_size=256, cellpose_channels=[0, 1, None], gpu=True),
    n_jobs=4,
)
well_profile = orch.run(batch)   # DataFrame: grouping keys + all feature columns
```

`n_jobs` controls joblib parallelism across cells/fields in the per-cell and
field+mask extractors.

## Illumination correction

`IlluminationCorrection` implements the retrospective multi-image method
(Caicedo et al. 2017): fit one smooth, unit-mean illumination function per
group (plate) per channel, then divide. Apply it to the engineered-feature
path (segmentation, structure tensor, cp_measure) but **not** to the
autoencoder, which expects its own pretraining preprocessing.

```python
from morphology_featurization_tools import IlluminationCorrection
ic = IlluminationCorrection(smoothing_sigma=100.0)
ic.fit(fields, plate_labels)
corrected = ic.transform(fields, plate_labels)
```

## Batch correction and mAP

Well profiles are normalized against negative controls with
`TypicalVariationNormalization` or `MADRobustize` (`normalizers.py`) and scored
with copairs (`retrieval.py`).

## Diagnostics

`plate_layout_heatmap` and `control_correlation_matrix` reproduce Caicedo
Figure 4 (plate-layout effects and batch effects).

## Optional dependencies

- `cp-measure` — CellProfiler-style `cp_*` features (`pip install cp-measure`)
- `copairs` — mAP retrieval evaluation (`pip install copairs`)
- `cellpose>=4`, `jump-portrait` — segmentation and JUMP image fetching
- OpenPhenom model code — `autoencoder` features; downloaded by
  `install_openphenom.sh` at the repository root after you accept its license

`cp_measure` and `copairs` are imported lazily; the package loads without them.
