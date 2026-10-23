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


def read_image(image):
    """Read an image into a 2-D float array with intensities in [0, 1].

    Parameters
    ----------
    image : str or numpy.ndarray
        File path to an image (supported file formats are .png, .jpeg,
        .jpg, and .tiff) or an array of pixel intensities. Arrays with
        more than 2 dimensions are reduced to their first channel.

    Returns
    -------
    numpy.ndarray
        2-D array of pixel intensities.
    """
    if isinstance(image, np.ndarray):
        image_data = image
    else:
        import imageio.v3 as iio

        image_data = iio.imread(image)
    if image_data.ndim > 2:
        image_data = image_data[:, :, 0]
    image_data = np.asarray(image_data)
    if np.issubdtype(image_data.dtype, np.integer):
        image_data = image_data.astype(np.float64) / np.iinfo(image_data.dtype).max
    else:
        image_data = image_data.astype(np.float64)
    return image_data


def low_pixel_threshold(image, threshold_percentage):
    """Set the lowest pixels (by percentage of maximum intensity) to 0.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array of pixel intensities.
    threshold_percentage : float
        A number between 0 and 1. Pixels with intensity below
        ``threshold_percentage`` times the maximum intensity are set to 0.

    Returns
    -------
    numpy.ndarray
        Thresholded copy of the image.
    """
    image = image.copy()
    intensity_threshold = image.max() * threshold_percentage
    image[image < intensity_threshold] = 0.0
    return image


def create_cdf_vector(image):
    """Cumulative sum of pixel intensities in column-major order.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array of pixel intensities.

    Returns
    -------
    numpy.ndarray
        1-D array of cumulative intensity sums.
    """
    return np.cumsum(image.flatten(order="F"))


def create_disc_mask(radius):
    """Boolean disc mask of the given pixel radius.

    Parameters
    ----------
    radius : float
        Radius of the disc in pixels.

    Returns
    -------
    numpy.ndarray
        Square boolean array where entries within ``radius`` of the
        center are True.
    """
    size = int(2 * radius + 1)
    center = math.ceil(size / 2)
    rows = np.arange(1, size + 1)[:, None]
    cols = np.arange(1, size + 1)[None, :]
    distances = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    return distances <= radius


class Patch:
    """A disc-masked patch extracted from an image.

    Attributes
    ----------
    data : numpy.ndarray or None
        Disc-masked pixel intensities of the patch.
    center : numpy.ndarray
        (row, column) indices of the patch center in the source image.
    xmin, xmax : int
        Row extent of the patch in the source image, as a half-open
        interval ``[xmin, xmax)``.
    ymin, ymax : int
        Column extent of the patch in the source image, as a half-open
        interval ``[ymin, ymax)``.
    sample : numpy.ndarray or None
        Point sample of the patch, when retained.
    """

    __slots__ = ("data", "center", "xmin", "xmax", "ymin", "ymax", "sample")

    def __init__(self, data, center, xmin, xmax, ymin, ymax, sample=None):
        self.data = data
        self.center = center
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.sample = sample


def get_patch_from_point_and_mask(image, point, mask):
    """Extract the disc-masked patch of an image around a point.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array of pixel intensities.
    point : sequence of int
        (row, column) indices of the patch center.
    mask : numpy.ndarray
        Disc mask from :func:`create_disc_mask`.

    Returns
    -------
    Patch
        The extracted patch.
    """
    radius = mask.shape[0] // 2
    n_rows, n_cols = image.shape
    patch_lower_x = max(point[0] - radius, 0)
    patch_upper_x = min(point[0] + radius, n_rows)
    patch_lower_y = max(point[1] - radius, 0)
    patch_upper_y = min(point[1] + radius, n_cols)

    rectangular_patch = image[patch_lower_x:patch_upper_x, patch_lower_y:patch_upper_y]
    width, length = rectangular_patch.shape
    data = rectangular_patch * mask[:width, :length]
    return Patch(
        data=data,
        center=np.asarray(point),
        xmin=patch_lower_x,
        xmax=patch_upper_x,
        ymin=patch_lower_y,
        ymax=patch_upper_y,
    )


def patch_sample_with_circ_boundary(sample, radius, epsilon):
    """Append an epsilon-spaced circular boundary point set to a sample.

    Parameters
    ----------
    sample : numpy.ndarray
        2 x n array of point coordinates.
    radius : float
        Radius of the circular boundary.
    epsilon : float
        Maximum spacing between consecutive boundary points.

    Returns
    -------
    numpy.ndarray
        2 x (n + m) array with the boundary points appended.
    """
    n = math.ceil(2 * math.pi / math.acos(1 - (epsilon**2 / (2 * radius**2))))
    angles = 2 * math.pi * np.arange(1, n + 1) / n
    x_coordinates = radius * np.cos(angles) + radius
    y_coordinates = radius * np.sin(angles) + radius
    circle_boundary = np.vstack([x_coordinates, y_coordinates])
    return np.hstack([np.asarray(sample, dtype=np.float64), circle_boundary])
