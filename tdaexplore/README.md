TDAExplore
-------------

Copyright (C) 2021 [Parker
Edwards](https://sites.nd.edu/parker-edwards)

Description
-----------

Python package for sampling local regions from images and computing a persistence landscape for each region.

Version 1.0.0
-------------

Installation
---------------------------
```
    pip install tdaexplore
```
To install from source:
```
    pip install .
```

Usage
------
After installation check `help(tdaexplore.patch_landscapes_from_image)` for starter examples.

The main exports are:

* `patch_landscapes_from_image`: Sample patches from a single image and compute a landscape vector for each patch.
* `PatchLandscapeTransformer`: A scikit-learn compatible transformer which applies the same featurization to collections of image arrays.
* `compute_landscapes`, `run_svm`, `run_multisvm`, `run_random_forest`: Pipelines which featurize directories of images and cross-validate classifiers on the results. Parameters may be passed directly or through a parameters csv file; directly passed parameters take precedence.
* `compute_image_weights`, `plot_image_mask`: Convolve an image with patch scores and plot the resulting mask.
* `demonstrate_image`: Display where patches are sampled from an image and the point samples computed from them.

License
-------

TDAExplore is licensed under GPLv3.
