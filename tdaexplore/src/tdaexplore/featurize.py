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

import math

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .image import (
    create_cdf_vector,
    create_disc_mask,
    get_patch_from_point_and_mask,
    low_pixel_threshold,
    patch_sample_with_circ_boundary,
    read_image,
)
from .landscapes import landscapes_from_samples
from .sampling import sample_centers_using_cdf_sparse, sample_patch_using_top_intensities_sparse

DELTA = 2.5


def patch_landscapes_from_image(
    image,
    number_of_patches,
    pixel_radius_for_patches=50,
    number_of_cores=1,
    patch_center_image=None,
    return_patches=False,
    remove_patch_data=True,
    return_samples=False,
    lower_threshold=0,
    upper_threshold=1,
    proportion_of_patch_sparse=0.025,
    noise_threshold=0.05,
    number_of_layers=50,
    random_state=None,
):
    """Sample patches from an image and compute landscapes from them.

    The main featurization function of TDAExplore for programmatic
    access. Most users will only be interested in the first 4
    parameters. A few parameters are undocumented and it is inadvisable
    to change them.

    Parameters
    ----------
    image : str or numpy.ndarray
        File path (absolute or relative to current directory) to the
        image, or a 2-D array of pixel intensities. Supported file
        formats are .png, .jpeg, .jpg, and .tiff.
    number_of_patches : int
        Number of patches to sample from image.
    pixel_radius_for_patches : int
        All sampled patches will be of this radius, default is 50
        pixels.
    number_of_cores : int
        The number of CPU cores to parallelize over when computing
        landscapes for this image. Default is 1 (no parallelization).
    patch_center_image : str or numpy.ndarray, optional
        An image to use for selecting the centers of patches, given in
        the same forms as ``image``. Useful if you have a masked
        version of the image where you want patches to be centered.
        Defaults to None, in which case ``image`` is used for patch
        centers.
    return_patches : bool
        If set to True, output is a dictionary
        ``image_landscape_results`` where
        ``image_landscape_results["data"]`` is the same data matrix as
        the default output and ``image_landscape_results["patches"]``
        is a list of patch objects (in the same order as the rows of
        the data matrix) that describe the center and position in the
        original image for each patch. Default is False, in which case
        output is described in Returns below.
    remove_patch_data : bool
        If set to False and ``return_patches`` is True, the patch
        details also include the raw subset of the image comprising
        each patch. Useful for some imaging tasks. WARNING: Memory
        costs can quickly get out of hand when setting this to True.
        Default is True, in which case such data is not retained.
    return_samples : bool
        If set to True and ``return_patches`` is True, will also append
        the samples to each patch entry. Default is False. Useful for
        some imaging tasks. WARNING: Memory costs can quickly get out
        of hand when setting this to True.
    noise_threshold : float
        A number between 0 and 1 representing a percentage. The bottom
        ``noise_threshold`` pixels in the image by intensity will be
        set to 0. Used to eliminate small intensity background noise.
        Default is .05, i.e. 5%.
    number_of_layers : int
        For TDA experts: the maximum landscape Lambda_{number_of_layers}
        to compute. Default is 50.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for patch center sampling. Default is None,
        in which case fresh entropy is used.

    Returns
    -------
    scipy.sparse.csr_matrix or dict
        Returns a sparse matrix ``image_landscape_results`` with
        ``number_of_patches`` rows. ``image_landscape_results[i, :]``
        is the landscape vector computed for the i'th sampled patch.
        If ``return_patches`` is True, instead returns a dictionary
        with entries "data", "patches", "centroid", and "centers".
    """
    rng = np.random.default_rng(random_state)
    spacing_for_disc_points = DELTA / 2

    image_data = read_image(image)
    image_data = low_pixel_threshold(image_data, noise_threshold)

    if patch_center_image is not None and patch_center_image is not False:
        center_image = read_image(patch_center_image)
        center_image = low_pixel_threshold(center_image, noise_threshold)
        image_cdf = create_cdf_vector(center_image)
        sampling_dims = center_image.shape
    else:
        image_cdf = create_cdf_vector(image_data)
        sampling_dims = image_data.shape

    center_points = sample_centers_using_cdf_sparse(
        image_cdf, number_of_patches, sampling_dims, pixel_radius_for_patches / 4, rng
    )
    on_pixels = np.argwhere(image_data != 0)
    centroid = on_pixels.mean(axis=0)

    radial_thresholding = (upper_threshold - lower_threshold) < 1
    if radial_thresholding:
        distances_from_centroid = np.linalg.norm(on_pixels - centroid, axis=1)
        estimated_radius = distances_from_centroid.max()
        center_distances = np.linalg.norm(center_points.T - centroid, axis=1) / estimated_radius
        valid = (center_distances >= lower_threshold) & (center_distances <= upper_threshold)
        center_points = center_points[:, valid]

    # Forces sampling up to required number of patches, gradually
    # relaxing sampling restriction
    k = 3
    while center_points.shape[1] < number_of_patches:
        new_points = sample_centers_using_cdf_sparse(
            image_cdf, number_of_patches, sampling_dims, pixel_radius_for_patches / (2**k), rng
        )
        if radial_thresholding:
            center_distances = np.linalg.norm(new_points.T - centroid, axis=1) / estimated_radius
            valid = (center_distances >= lower_threshold) & (center_distances <= upper_threshold)
            new_points = new_points[:, valid]
        center_points = np.hstack([center_points, new_points])
        k += 1

    center_points = center_points[:, :number_of_patches]

    mask = create_disc_mask(pixel_radius_for_patches)
    patches = [
        get_patch_from_point_and_mask(image_data, center_points[:, i], mask)
        for i in range(center_points.shape[1])
    ]

    samples = []
    exclusion_disc = create_disc_mask(DELTA)
    for patch in patches:
        sample = sample_patch_using_top_intensities_sparse(patch, proportion_of_patch_sparse, exclusion_disc)
        samples.append(patch_sample_with_circ_boundary(sample, pixel_radius_for_patches, spacing_for_disc_points))

    if return_patches and return_samples:
        for patch, sample in zip(patches, samples):
            patch.sample = sample
    if remove_patch_data:
        for patch in patches:
            patch.data = None

    rval = landscapes_from_samples(
        samples, pixel_radius_for_patches, number_of_cores=number_of_cores, number_of_layers=number_of_layers
    )

    if return_patches:
        return {"data": rval, "patches": patches, "centroid": centroid, "centers": center_points}
    return rval


class PatchLandscapeTransformer(TransformerMixin, BaseEstimator):
    """Transform images into matrices of patch landscape vectors.

    Samples patches from each input image and computes a persistence
    landscape vector for each patch. Each image contributes
    ``number_of_patches`` rows to the output matrix, in input order.

    Parameters
    ----------
    number_of_patches : int
        Number of patches to sample from each image.
    pixel_radius_for_patches : int
        All sampled patches will be of this radius, default is 50
        pixels.
    proportion_of_patch_sparse : float
        Proportion of each patch's nonzero pixels retained by the
        sparse point sampler. Default is .025.
    noise_threshold : float
        A number between 0 and 1 representing a percentage. The bottom
        ``noise_threshold`` pixels in each image by intensity will be
        set to 0. Default is .05.
    lower_threshold, upper_threshold : float
        Restrict patch centers to a radial band around the image
        centroid, as a fraction of the estimated image radius. Defaults
        of 0 and 1 place no restriction.
    number_of_layers : int
        The maximum landscape Lambda_{number_of_layers} to compute per
        homological degree. Default is 50.
    n_jobs : int
        Number of images to featurize in parallel. Default is 1.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for patch center sampling. Default is None,
        in which case fresh entropy is used.

    Attributes
    ----------
    patches_per_image_ : int
        Number of rows contributed by each input image, set after
        ``transform``.
    """

    def __init__(
        self,
        number_of_patches=None,
        pixel_radius_for_patches=50,
        proportion_of_patch_sparse=0.025,
        noise_threshold=0.05,
        lower_threshold=0,
        upper_threshold=1,
        number_of_layers=50,
        patch_ratio=2,
        n_jobs=1,
        random_state=None,
    ):
        self.number_of_patches = number_of_patches
        self.pixel_radius_for_patches = pixel_radius_for_patches
        self.proportion_of_patch_sparse = proportion_of_patch_sparse
        self.noise_threshold = noise_threshold
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.number_of_layers = number_of_layers
        self.patch_ratio = patch_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """No-op fit for scikit-learn compatibility.

        Parameters
        ----------
        X : sequence of numpy.ndarray or numpy.ndarray
            List of 2-D image arrays, or a 3-D array of images.
        y : ignored

        Returns
        -------
        PatchLandscapeTransformer
            The transformer instance.
        """
        return self

    def _patches_for_image(self, image):
        if self.number_of_patches is not None:
            return self.number_of_patches
        pixel_area = image.shape[0] * image.shape[1]
        return max(
            math.floor(self.patch_ratio * pixel_area / (math.pi * self.pixel_radius_for_patches**2)), 2
        )

    def transform(self, X):
        """Compute patch landscape vectors for a collection of images.

        Parameters
        ----------
        X : sequence of numpy.ndarray or numpy.ndarray
            List of 2-D image arrays, or a 3-D array whose first axis
            indexes images.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix with one row per sampled patch. Rows are grouped by
            image in input order, ``patches_per_image_`` rows per
            image.
        """
        images = [np.asarray(image) for image in X]
        if len(images) == 0:
            raise ValueError("X contains no images.")
        patches_per_image = self._patches_for_image(images[0])

        rng = np.random.default_rng(self.random_state)
        seeds = rng.spawn(len(images)) if self.random_state is not None else [None] * len(images)

        def featurize(image, seed):
            return patch_landscapes_from_image(
                image,
                patches_per_image,
                pixel_radius_for_patches=self.pixel_radius_for_patches,
                proportion_of_patch_sparse=self.proportion_of_patch_sparse,
                noise_threshold=self.noise_threshold,
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                number_of_layers=self.number_of_layers,
                random_state=seed,
            )

        if self.n_jobs is not None and self.n_jobs != 1:
            from joblib import Parallel, delayed

            blocks = Parallel(n_jobs=self.n_jobs)(
                delayed(featurize)(image, seed) for image, seed in zip(images, seeds)
            )
        else:
            blocks = [featurize(image, seed) for image, seed in zip(images, seeds)]

        self.patches_per_image_ = patches_per_image
        return sparse.vstack(blocks, format="csr")
