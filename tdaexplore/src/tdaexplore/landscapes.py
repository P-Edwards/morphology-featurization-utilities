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

import numpy as np
from scipy import sparse


def alpha_complex_diagrams(points):
    """Compute degree 0 and degree 1 alpha complex persistence diagrams.

    Filtration values are square-rooted so that birth and death values
    are in the same units as point distances. The death value of the
    essential degree 0 class is set to the maximum filtration value.

    Parameters
    ----------
    points : numpy.ndarray
        n x 2 array of point coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        (degree 0 pairs, degree 1 pairs), each a k x 2 array of
        (birth, death) values.
    """
    import gudhi

    alpha_complex = gudhi.AlphaComplex(points=np.asarray(points, dtype=np.float64))
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.compute_persistence()
    max_filtration = 0.0
    for _, filtration_value in simplex_tree.get_filtration():
        if filtration_value > max_filtration:
            max_filtration = filtration_value
    diagrams = []
    for degree in (0, 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(degree)
        if pairs.size == 0:
            diagrams.append(np.empty((0, 2)))
            continue
        pairs = pairs.copy()
        pairs[np.isinf(pairs[:, 1]), 1] = max_filtration
        diagrams.append(np.sqrt(pairs))
    return diagrams[0], diagrams[1]


def landscape_grid(min_x, max_x, dx):
    """Grid of x values used for discrete landscape computation.

    Parameters
    ----------
    min_x : float
        Left endpoint of the domain.
    max_x : float
        Right endpoint of the domain.
    dx : float
        Domain grid diameter. Grid points are spaced dx/2 apart.

    Returns
    -------
    numpy.ndarray
        1-D array of grid x values.
    """
    number_of_bins = int(2 * ((max_x - min_x) / dx) + 1)
    return min_x + 0.5 * dx * np.arange(number_of_bins)


def discrete_landscape(diagram, min_x, max_x, dx):
    """Discrete persistence landscape of a diagram on a grid.

    The value of the k'th landscape layer at grid point x is the k'th
    largest of min(x - birth, death - x) over the diagram's (birth,
    death) pairs with birth < x < death, or 0 if there are fewer than
    k such pairs.

    Parameters
    ----------
    diagram : numpy.ndarray
        k x 2 array of (birth, death) pairs.
    min_x : float
        Left endpoint of the domain.
    max_x : float
        Right endpoint of the domain.
    dx : float
        Domain grid diameter. Grid points are spaced dx/2 apart.

    Returns
    -------
    numpy.ndarray
        Array of shape (number of nonzero layers, number of grid
        points) with landscape values.
    """
    x_values = landscape_grid(min_x, max_x, dx)
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.size == 0:
        return np.zeros((0, x_values.size))
    births = diagram[:, 0][:, None]
    deaths = diagram[:, 1][:, None]
    x_row = x_values[None, :]
    values = np.minimum(x_row - births, deaths - x_row)
    values[(x_row <= births) | (x_row >= deaths)] = 0.0
    values = -np.sort(-values, axis=0)
    nonzero_layers = np.flatnonzero(values.any(axis=1))
    max_nonzero_lambda = nonzero_layers[-1] + 1 if nonzero_layers.size else 0
    return values[:max_nonzero_lambda]


def landscape_to_vector(landscape_degree_0, landscape_degree_1, number_of_layers=50):
    """Flatten a pair of discrete landscapes into a single vector.

    For each degree, up to ``number_of_layers`` layers are flattened in
    layer-major order into a block of length (number of layers) x
    (number of grid points), zero-padded if fewer layers are present.

    Parameters
    ----------
    landscape_degree_0 : numpy.ndarray
        Discrete landscape for degree 0.
    landscape_degree_1 : numpy.ndarray
        Discrete landscape for degree 1.
    number_of_layers : int
        Number of landscape layers retained per degree.

    Returns
    -------
    numpy.ndarray
        1-D landscape vector.
    """
    blocks = []
    for landscape in (landscape_degree_0, landscape_degree_1):
        n_grid = landscape.shape[1]
        block = np.zeros(n_grid * number_of_layers)
        layers = min(number_of_layers, landscape.shape[0])
        if layers > 0:
            block[: layers * n_grid] = landscape[:layers].ravel()
        blocks.append(block)
    return np.concatenate(blocks)


def landscapes_from_samples(samples, pixel_radius_for_patches, number_of_cores=1, number_of_layers=50):
    """Compute a sparse matrix of landscape vectors from point samples.

    Each sample is passed through alpha complex persistence in degrees
    0 and 1, discrete landscapes are computed on a grid of diameter
    ``pixel_radius_for_patches / 200`` over
    [0, ``pixel_radius_for_patches``], and the resulting vectors are
    collected as rows of a sparse matrix.

    Parameters
    ----------
    samples : list of numpy.ndarray
        List of 2 x n point samples.
    pixel_radius_for_patches : float
        Radius of the sampled patches in pixels.
    number_of_cores : int
        Number of CPU cores to parallelize over when computing
        persistence diagrams.
    number_of_layers : int
        Number of landscape layers retained per degree.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix with one landscape vector per sample.
    """
    if number_of_cores > 1:
        from joblib import Parallel, delayed

        diagrams = Parallel(n_jobs=number_of_cores)(
            delayed(alpha_complex_diagrams)(np.asarray(sample).T) for sample in samples
        )
    else:
        diagrams = [alpha_complex_diagrams(np.asarray(sample).T) for sample in samples]

    dx = pixel_radius_for_patches / 200
    rows = []
    for diagram_degree_0, diagram_degree_1 in diagrams:
        landscape_degree_0 = discrete_landscape(diagram_degree_0, 0, pixel_radius_for_patches, dx)
        landscape_degree_1 = discrete_landscape(diagram_degree_1, 0, pixel_radius_for_patches, dx)
        rows.append(
            sparse.csr_matrix(landscape_to_vector(landscape_degree_0, landscape_degree_1, number_of_layers))
        )
    return sparse.vstack(rows, format="csr")
