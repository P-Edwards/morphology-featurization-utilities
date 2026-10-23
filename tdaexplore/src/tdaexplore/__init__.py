# Python Package TDAExplore

# Copyright (C) 2021 Parker Edwards

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

"""Image analysis using persistence landscapes.

Subsamples local regions ("patches") from an input image, generates
landscapes from them, and outputs landscapes in a matrix format.
"""

from .featurize import PatchLandscapeTransformer, patch_landscapes_from_image
from .image import Patch, read_image
from .params import read_parameters
from .landscapes import (
    alpha_complex_diagrams,
    discrete_landscape,
    landscape_grid,
    landscape_to_vector,
    landscapes_from_samples,
)
from .pipelines import (
    average_vectors_for_images,
    compute_landscapes,
    heuristic_C,
    run_multisvm,
    run_random_forest,
    run_svm,
)
from .plotting import compute_image_weights, demonstrate_image, plot_image_mask

weight_image_using_landscapes_and_transform = compute_image_weights

__version__ = "1.0.0"

__all__ = [
    "PatchLandscapeTransformer",
    "patch_landscapes_from_image",
    "Patch",
    "read_image",
    "read_parameters",
    "alpha_complex_diagrams",
    "discrete_landscape",
    "landscape_grid",
    "landscape_to_vector",
    "landscapes_from_samples",
    "average_vectors_for_images",
    "compute_landscapes",
    "heuristic_C",
    "run_svm",
    "run_multisvm",
    "run_random_forest",
    "compute_image_weights",
    "weight_image_using_landscapes_and_transform",
    "plot_image_mask",
    "demonstrate_image",
]
