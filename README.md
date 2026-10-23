# morphology_featurization_tools

Featurization of microscopy images for morphological profiling: streaming
datasets over the JUMP Cell Painting images (cpg0016), per-plate illumination
correction, Cellpose segmentation, CellProfiler-style features (cp_measure),
OpenPhenom autoencoder embeddings, structure-tensor and TDAExplore
persistence-landscape features, and normalization and retrieval scoring of
well profiles.

## Installation

Requirements: [conda](https://docs.conda.io/) (Miniconda or Anaconda). An
NVIDIA GPU is recommended.

```bash
git clone <repository URL>
cd <repository directory>
./install.sh
conda activate ./taxane_conda_env
```

`install.sh` creates the environment described by `environment.yaml` in
`./taxane_conda_env`, or updates it if it already exists. Set the environment variable
`TAXANE_ENV_PREFIX` to create it elsewhere. The environment can also be
created directly with `conda env create --file environment.yaml --prefix <directory>`.

`environment.yaml` installs PyTorch 2.5.0 built for CUDA 12.1 on Linux. On
other platforms (macOS, or Linux without CUDA 12.1 support), replace the
three PyTorch lines under `pip:` with the PyTorch 2.5 install for that
platform before running `install.sh`.

The environment installs the three packages of this repository in editable
mode:

- `morphology_featurization_tools`: datasets, preprocessing, segmentation,
  feature extractors, normalizers and retrieval scoring.
- `openphenom`: a placeholder for the model code of Recursion's OpenPhenom
  masked autoencoder, which is installed separately (below).
- `tdaexplore`: TDAExplore persistence-landscape features.

### Optional: OpenPhenom autoencoder

The autoencoder features use
[OpenPhenom](https://huggingface.co/recursionpharma/OpenPhenom), whose code
and weights are distributed under Recursion's Non-Commercial End User License
Agreement rather than the license of this repository, so they are not
included. To use them, run

```bash
./install_openphenom.sh
```

It links to the license, asks you to accept it, and downloads the model code
from a fixed revision of the Hugging Face repository into
`openphenom/openphenom/`, checking each file. In scripts, pass
`--accept-license` once you have read the license. The pretrained weights are
downloaded from Hugging Face the first time the autoencoder runs. Everything
else in the package works without OpenPhenom.

JUMP images are read from the public Cell Painting Gallery S3 bucket, with
no AWS account needed. The JUMP metadata tables and image index are downloaded
on first use and cached.

## Getting started

The notebook in [`notebooks/`](notebooks/) shows the package in use:

- [`target2_featurization.ipynb`](notebooks/target2_featurization.ipynb)
  downloads a few wells of a JUMP Target-2 plate, displays a field, computes
  CellProfiler-style well profiles with illumination correction and Cellpose
  segmentation, and computes OpenPhenom autoencoder well profiles.


The feature families and their interfaces are described in
[`morphology_featurization_tools/FEATURES.md`](morphology_featurization_tools/FEATURES.md).
