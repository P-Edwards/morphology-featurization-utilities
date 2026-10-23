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

from .image import create_disc_mask


def sparse_top_intensities(pixels_in_order, number_of_pixels_to_sample, data_rows, data_cols, exclusion_disc_data):
    """Greedily select pixels in the given order subject to a spacing constraint.

    Pixels are visited in order and accepted if no previously accepted
    pixel's exclusion disc covers them.

    Parameters
    ----------
    pixels_in_order : numpy.ndarray
        1-D array of candidate pixel indices, flattened in column-major
        order.
    number_of_pixels_to_sample : int
        Maximum number of pixels to accept.
    data_rows, data_cols : int
        Dimensions of the source array.
    exclusion_disc_data : numpy.ndarray
        Boolean disc mask stamped around each accepted pixel.

    Returns
    -------
    numpy.ndarray
        2 x k array of (row, column) indices of accepted pixels, with
        k at most ``number_of_pixels_to_sample``.
    """
    proximity_mask = np.zeros((data_rows, data_cols), dtype=bool)
    sampled = np.zeros((2, max(number_of_pixels_to_sample, 1)), dtype=np.int64)
    n_sampled = 0

    disc_length, disc_width = exclusion_disc_data.shape
    middle_point_1 = math.ceil(disc_length / 2)
    middle_point_2 = math.ceil(disc_width / 2)

    for pixel in pixels_in_order:
        if n_sampled >= number_of_pixels_to_sample:
            break
        if pixel < 1:
            continue
        index_1 = (pixel - 1) % data_rows
        index_2 = (pixel - 1) // data_rows
        if not proximity_mask[index_1, index_2]:
            sampled[0, n_sampled] = index_1
            sampled[1, n_sampled] = index_2
            n_sampled += 1
            image_row_minimum = max(index_1 - middle_point_1, 0)
            image_row_maximum = min(index_1 + (disc_length - middle_point_1), data_rows)
            image_col_minimum = max(index_2 - middle_point_2, 0)
            image_col_maximum = min(index_2 + (disc_width - middle_point_2), data_cols)
            disc_row_minimum = middle_point_1 - (index_1 - image_row_minimum)
            disc_row_maximum = middle_point_1 + (image_row_maximum - index_1)
            disc_col_minimum = middle_point_2 - (index_2 - image_col_minimum)
            disc_col_maximum = middle_point_2 + (image_col_maximum - index_2)
            rows = slice(image_row_minimum + disc_row_minimum, image_row_minimum + disc_row_maximum)
            cols = slice(image_col_minimum + disc_col_minimum, image_col_minimum + disc_col_maximum)
            np.logical_or(
                proximity_mask[rows, cols],
                exclusion_disc_data[disc_row_minimum:disc_row_maximum, disc_col_minimum:disc_col_maximum],
                out=proximity_mask[rows, cols],
            )
    if n_sampled == 0:
        return np.zeros((2, 1), dtype=np.int64)
    return sampled[:, :n_sampled]


def sample_patch_using_top_intensities_sparse(patch, proportion_of_patch_sparse=0.025, exclusion_disc_data=None):
    """Sample a spaced subset of a patch's highest-intensity pixels.

    Parameters
    ----------
    patch : Patch
        Patch with intensity data.
    proportion_of_patch_sparse : float
        Proportion of the patch's nonzero pixels to sample.
    exclusion_disc_data : numpy.ndarray, optional
        Boolean disc mask enforcing spacing between sampled pixels.
        Defaults to a disc of radius 2.5.

    Returns
    -------
    numpy.ndarray
        2 x k array of (row, column) indices of sampled pixels.
    """
    if exclusion_disc_data is None:
        exclusion_disc_data = create_disc_mask(2.5)
    data = patch.data
    nonzero_pixels = int(np.count_nonzero(data))
    number_of_pixels_to_sample = math.ceil(nonzero_pixels * proportion_of_patch_sparse)
    flattened = data.flatten(order="F")
    pixels_in_order = np.argsort(-flattened, kind="stable")[:nonzero_pixels] + 1
    return sparse_top_intensities(
        pixels_in_order, number_of_pixels_to_sample, data.shape[0], data.shape[1], exclusion_disc_data
    )


def sample_centers_using_cdf_sparse(image_cdf, number_of_pixels_to_sample, dimensions, delta, rng):
    """Sample spaced pixels with probability proportional to intensity.

    Draws candidate pixels using the inverse of the image's intensity
    CDF, then greedily accepts candidates subject to a ``delta``-radius
    spacing constraint.

    Parameters
    ----------
    image_cdf : numpy.ndarray
        Cumulative intensity vector from :func:`create_cdf_vector`.
    number_of_pixels_to_sample : int
        Maximum number of pixels to accept.
    dimensions : tuple of int
        Dimensions of the source image.
    delta : float
        Radius of the exclusion disc between accepted pixels.
    rng : numpy.random.Generator
        Random generator used to draw candidates.

    Returns
    -------
    numpy.ndarray
        2 x k array of (row, column) indices of sampled pixels.
    """
    exclusion_disc_data = create_disc_mask(delta)
    random_samples = rng.uniform(0.0, image_cdf[-1], 3 * number_of_pixels_to_sample)
    potential_samples = np.searchsorted(image_cdf, random_samples, side="left")
    return sparse_top_intensities(
        potential_samples, number_of_pixels_to_sample, dimensions[0], dimensions[1], exclusion_disc_data
    )
