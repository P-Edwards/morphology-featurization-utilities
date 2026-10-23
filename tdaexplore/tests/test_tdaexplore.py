import os

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

import tdaexplore as tda
from tdaexplore.image import create_disc_mask, read_image
from tdaexplore.landscapes import discrete_landscape, landscape_grid
from tdaexplore.sampling import sparse_top_intensities

TRIANGLES = os.path.join(os.path.dirname(__file__), "data", "triangles.jpg")


def test_create_disc_mask():
    mask = create_disc_mask(2.5)
    assert mask.shape == (6, 6)
    assert mask[2, 2]
    assert not mask[0, 0]


def test_sparse_top_intensities_spacing():
    disc = create_disc_mask(2.5)
    pixels = np.arange(1, 101)
    sampled = sparse_top_intensities(pixels, 10, 10, 10, disc)
    assert sampled.shape[0] == 2
    for i in range(sampled.shape[1]):
        for j in range(i + 1, sampled.shape[1]):
            assert not np.array_equal(sampled[:, i], sampled[:, j])


def test_discrete_landscape_values():
    diagram = np.array([[0.0, 4.0], [1.0, 3.0]])
    landscape = discrete_landscape(diagram, 0, 4, 1.0)
    grid = landscape_grid(0, 4, 1.0)
    assert grid.size == 9
    assert np.allclose(landscape[0], [0, 0.5, 1, 1.5, 2, 1.5, 1, 0.5, 0])
    assert np.allclose(landscape[1], [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0])


def test_discrete_landscape_empty():
    landscape = discrete_landscape(np.empty((0, 2)), 0, 4, 1.0)
    assert landscape.shape == (0, 9)


def test_patch_landscapes_from_image():
    result = tda.patch_landscapes_from_image(TRIANGLES, 8, pixel_radius_for_patches=40, random_state=0)
    assert sparse.issparse(result)
    assert result.shape == (8, 2 * 50 * 401)
    assert result.nnz > 0


def test_patch_landscapes_return_patches():
    result = tda.patch_landscapes_from_image(
        TRIANGLES,
        4,
        pixel_radius_for_patches=40,
        return_patches=True,
        remove_patch_data=False,
        return_samples=True,
        random_state=0,
    )
    assert result["data"].shape[0] == 4
    assert len(result["patches"]) == 4
    patch = result["patches"][0]
    assert patch.data is not None
    assert patch.sample.shape[0] == 2
    assert result["centers"].shape == (2, 4)


def test_transformer_shapes_and_seed():
    image = read_image(TRIANGLES)
    transformer = tda.PatchLandscapeTransformer(
        number_of_patches=5, pixel_radius_for_patches=40, random_state=11
    )
    X = transformer.fit_transform([image, image])
    assert X.shape == (10, 2 * 50 * 401)
    assert transformer.patches_per_image_ == 5
    repeat = clone(transformer).fit_transform([image, image])
    assert (X != repeat).nnz == 0


def test_transformer_tensor_input():
    image = read_image(TRIANGLES)
    tensor = np.stack([image, image])
    X = tda.PatchLandscapeTransformer(number_of_patches=3, pixel_radius_for_patches=40).fit_transform(tensor)
    assert X.shape[0] == 6


def test_compute_image_weights_array_input():
    image = read_image(TRIANGLES)
    result = tda.patch_landscapes_from_image(
        image, 6, pixel_radius_for_patches=40, return_patches=True, random_state=0
    )
    weights = tda.compute_image_weights(
        image,
        result["data"],
        result["patches"],
        40,
        transformation_function=lambda M: np.asarray(M.sum(axis=1)).ravel(),
        interval_rep=True,
    )
    assert weights["mask"].shape == image.shape
    assert len(weights["weights"]) == 6
    assert len(weights["interval_rep"]) == 50


@pytest.fixture(scope="module")
def image_directories(tmp_path_factory):
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    import imageio.v3 as iio

    for class_name, blob_radius in (("a", 3), ("b", 10)):
        directory = root / class_name
        directory.mkdir()
        for i in range(4):
            image = np.zeros((120, 120))
            yy, xx = np.mgrid[0:120, 0:120]
            for _ in range(12):
                cx, cy = rng.integers(20, 100, 2)
                image += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * blob_radius**2)))
            iio.imwrite(directory / f"im{i}.png", (np.clip(image, 0, 1) * 255).astype(np.uint8))
    return str(root / "a"), str(root / "b")


def test_pipelines(image_directories, tmp_path):
    landscapes = tda.compute_landscapes(
        image_directories=list(image_directories),
        directory_classes=["a", "b"],
        radius_of_patches=20,
        patch_ratio=0.5,
        data_results_directory=str(tmp_path / "results"),
        number_of_cores=1,
        random_state=0,
    )
    assert landscapes["summaries"].shape[0] == 8 * landscapes["patches_per_image"]
    assert landscapes["class_names"] == ["a", "b"]

    svm_results = tda.run_svm(landscapes=landscapes, number_of_folds=2, random_state=0)
    assert len(svm_results["svm"]) == 2
    assert svm_results["svm"][0]["svm_patch_accuracies"].shape == (2,)

    multisvm_results = tda.run_multisvm(landscapes=landscapes, number_of_folds=2, random_state=0)
    assert len(multisvm_results["multisvm"]) == 2

    forest_results = tda.run_random_forest(landscapes=landscapes, number_of_folds=2, random_state=0)
    assert len(forest_results["rand_forest"]) == 2


def test_parameters_file_and_overrides(image_directories, tmp_path):
    parameters = tmp_path / "parameters.csv"
    parameters.write_text(
        "image_directories,directory_classes,patch_center_images_directories,"
        "radius_of_patches,experiment_name,data_results_directory\n"
        f"{image_directories[0]},a,FALSE,30,test,{tmp_path / 'results'}\n"
        f"{image_directories[1]},b,FALSE,,,\n"
    )
    landscapes = tda.compute_landscapes(
        parameters=str(parameters), patch_ratio=0.5, radius_of_patches=20, random_state=0
    )
    assert landscapes["radius"] == 20
    assert landscapes["data_name_stem"].startswith("test")


def test_plot_image_mask():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = tda.patch_landscapes_from_image(
        TRIANGLES, 4, pixel_radius_for_patches=40, return_patches=True, random_state=0
    )
    weights = tda.compute_image_weights(TRIANGLES, np.arange(4, dtype=float), result["patches"], 40)
    ax = tda.plot_image_mask(weights)
    assert len(ax.images) == 2
    plt.close("all")


def test_demonstrate_image():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, info = tda.demonstrate_image(TRIANGLES, 4, pixel_radius_for_patches=40, random_state=0)
    assert len(figure.axes) == 1 + 2 * 4
    assert len(info["patches"]) == 4
    assert info["patches"][0].data is not None
    assert info["patches"][0].sample is not None
    plt.close(figure)
