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

from .featurize import patch_landscapes_from_image
from .image import create_disc_mask, low_pixel_threshold, read_image


def compute_image_weights(
    image,
    landscape_data,
    patches,
    radius,
    transformation_function=None,
    min_weight=0,
    max_weight=1,
    interval_rep=False,
):
    """Convolve an image using patch scores.

    Combines landscape data, data for positioning patches on an image,
    and a scoring function for landscapes to produce a mask for the
    image. A few parameters are undocumented and it is inadvisable to
    change them.

    Parameters
    ----------
    image : str or numpy.ndarray
        File path (absolute or relative to current directory) to the
        image, or a 2-D array of pixel intensities. Supported file
        formats are .png, .jpeg, .jpg, and .tiff.
    landscape_data : numpy.ndarray or scipy.sparse matrix
        Landscapes for image patches e.g. like those returned from
        ``patch_landscapes_from_image(return_patches=True)["data"]``.
    patches : sequence of Patch
        Patches e.g. like those returned from
        ``patch_landscapes_from_image(return_patches=True)["patches"]``.
    radius : int
        Radius of patches in pixels.
    transformation_function : callable, optional
        A function which takes as input a matrix and outputs a vector
        with scores, one for each row in the matrix. Defaults to the
        identity.
    interval_rep : bool
        If True, computes the interval representation described below.
        Default is False.

    Returns
    -------
    dict
        Returns a dictionary return_list with sufficient information
        to plot the image and its mask. return_list["image"] is a
        matrix with pixel intensities as values. return_list["mask"] is
        a matrix of the same dimensions with score intensities as
        values. return_list["interval_rep"] is a vector of length 50
        which compresses the mask into a radial representation.
        return_list["weights"] is a vector of the patch scores.
    """
    image_data = read_image(image)
    image_data = low_pixel_threshold(image_data, 0.05)
    if transformation_function is None:
        patch_weights = np.asarray(landscape_data)
    else:
        patch_weights = np.asarray(transformation_function(landscape_data))
    patch_weights = np.ravel(patch_weights).astype(np.float64)
    unmod_min_weight = patch_weights.min()
    unmod_max_weight = patch_weights.max()
    unmod_weights = patch_weights.copy()
    patch_weights = patch_weights - min_weight

    image_weight_mask = np.zeros_like(image_data)
    image_average_mask = np.ones_like(image_data)
    for patch in patches:
        image_average_mask[patch.xmin : patch.xmax, patch.ymin : patch.ymax] = 0
    for weight, patch in zip(patch_weights, patches):
        block = (slice(patch.xmin, patch.xmax), slice(patch.ymin, patch.ymax))
        image_weight_mask[block] += weight
        image_average_mask[block] += max_weight - min_weight
    total_weight_mask = image_weight_mask / image_average_mask
    total_weight_mask[image_data == 0] = 0

    if interval_rep:
        on_pixels = np.argwhere(image_data != 0)
        centroid = on_pixels.mean(axis=0)
        distances_from_centroid = np.linalg.norm(on_pixels - centroid, axis=1)
        estimated_radius = distances_from_centroid.max()
        rows = np.arange(image_data.shape[0])[:, None]
        cols = np.arange(image_data.shape[1])[None, :]
        percentage_mask = (
            np.sqrt((rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2) / estimated_radius
        )
        number_of_intervals = 50
        aggregated_output = np.zeros(number_of_intervals)
        for i in range(number_of_intervals):
            lower_threshold = i / number_of_intervals
            upper_threshold = (i + 1) / number_of_intervals
            active = (image_data > 0) & (percentage_mask >= lower_threshold) & (percentage_mask < upper_threshold)
            count = int(active.sum())
            aggregated_output[i] = total_weight_mask[active].sum() / count if count else np.nan
    else:
        aggregated_output = 0

    return {
        "image": image_data,
        "mask": total_weight_mask,
        "min": unmod_min_weight,
        "max": unmod_max_weight,
        "weights": unmod_weights,
        "interval_rep": aggregated_output,
    }


def plot_image_mask(weight_results, ax=None, cmap="RdBu_r", alpha=0.6):
    """Plot an image with its patch score mask overlaid.

    Parameters
    ----------
    weight_results : dict
        Results from :func:`compute_image_weights`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. Defaults to a new figure.
    cmap : str
        Colormap for the mask overlay.
    alpha : float
        Opacity of the mask overlay.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    image = weight_results["image"]
    mask = np.ma.masked_where(weight_results["mask"] == 0, weight_results["mask"])
    ax.imshow(image, cmap="gray", vmin=0, vmax=image.max())
    ax.imshow(mask, cmap=cmap, alpha=alpha)
    ax.set_axis_off()
    return ax


def demonstrate_image(
    image,
    number_of_patches,
    pixel_radius_for_patches=50,
    number_of_cores=1,
    proportion_of_patch_sparse=0.025,
    noise_threshold=0.05,
    number_of_layers=50,
    random_state=None,
    max_patches_displayed=None,
):
    """Display where patches are sampled from an image and what they contain.

    Produces a figure whose top panel shows the original image with
    translucent discs over patch locations, and whose lower panels
    show, for each patch, the patch's pixels alongside the point sample
    (with circular boundary) computed from it.

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
        landscapes for this image. Default is 1.
    proportion_of_patch_sparse : float
        Proportion of each patch's nonzero pixels retained by the
        sparse point sampler. Default is .025.
    noise_threshold : float
        A number between 0 and 1 representing a percentage. The bottom
        ``noise_threshold`` pixels in the image by intensity will be
        set to 0. Default is .05.
    number_of_layers : int
        The maximum landscape Lambda_{number_of_layers} to compute.
        Default is 50.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for patch center sampling. Default is None,
        in which case fresh entropy is used.
    max_patches_displayed : int, optional
        Maximum number of patches shown in the lower panels. Defaults
        to all patches.

    Returns
    -------
    tuple
        (figure, image_landscape_results) where figure is the
        matplotlib figure and image_landscape_results is the output of
        :func:`patch_landscapes_from_image` with patches, patch data,
        and samples attached.
    """
    import matplotlib.pyplot as plt

    base_information = patch_landscapes_from_image(
        image,
        number_of_patches,
        pixel_radius_for_patches=pixel_radius_for_patches,
        number_of_cores=number_of_cores,
        return_patches=True,
        remove_patch_data=False,
        return_samples=True,
        proportion_of_patch_sparse=proportion_of_patch_sparse,
        noise_threshold=noise_threshold,
        number_of_layers=number_of_layers,
        random_state=random_state,
    )

    original_image = read_image(image)
    patches = base_information["patches"]
    disc = create_disc_mask(pixel_radius_for_patches)
    mask = np.zeros_like(original_image, dtype=bool)
    for patch in patches:
        height = patch.xmax - patch.xmin
        width = patch.ymax - patch.ymin
        mask[patch.xmin : patch.xmax, patch.ymin : patch.ymax] |= disc[:height, :width]

    displayed = patches if max_patches_displayed is None else patches[:max_patches_displayed]
    n_columns = 4
    n_pairs_per_row = n_columns // 2
    n_rows = math.ceil(len(displayed) / n_pairs_per_row)

    figure = plt.figure(figsize=(3 * n_columns, 3 * (n_rows + n_pairs_per_row)))
    grid = figure.add_gridspec(n_rows + n_pairs_per_row, n_columns)

    overview_axes = figure.add_subplot(grid[:n_pairs_per_row, :])
    overview_axes.imshow(original_image, cmap="gray", vmin=0, vmax=original_image.max())
    overview_axes.imshow(
        np.ma.masked_where(~mask, mask.astype(float)), cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1
    )
    overview_axes.set_axis_off()

    for i, patch in enumerate(displayed):
        row = n_pairs_per_row + i // n_pairs_per_row
        column = 2 * (i % n_pairs_per_row)
        patch_axes = figure.add_subplot(grid[row, column])
        patch_axes.imshow(patch.data, cmap="gray", vmin=0, vmax=max(original_image.max(), 1e-12))
        patch_axes.set_axis_off()
        patch_axes.set_title(f"patch {i}", fontsize=8)

        sample_axes = figure.add_subplot(grid[row, column + 1])
        sample = patch.sample
        sample_axes.scatter(sample[1], sample[0], s=4)
        sample_axes.set_xlim(-1, 2 * pixel_radius_for_patches + 1)
        sample_axes.set_ylim(2 * pixel_radius_for_patches + 1, -1)
        sample_axes.set_aspect("equal")
        sample_axes.set_axis_off()
        sample_axes.set_title(f"sample {i}", fontsize=8)

    figure.tight_layout()
    return figure, base_information
