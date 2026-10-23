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
import os
import time
import warnings

import numpy as np
from scipy import sparse

from .featurize import patch_landscapes_from_image
from .image import read_image
from .params import read_parameters


def heuristic_C(data):
    """Heuristic cost parameter for a linear SVM.

    Computes the inverse of the mean squared Euclidean norm of the
    rows of ``data``.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse matrix
        Data matrix with one sample per row.

    Returns
    -------
    float
        Heuristic cost value.
    """
    if sparse.issparse(data):
        squared_norms = np.asarray(data.multiply(data).sum(axis=1)).ravel()
    else:
        data = np.asarray(data)
        squared_norms = (data * data).sum(axis=1)
    return 1.0 / squared_norms.mean()


def average_vectors_for_images(data_matrix, number_of_patches, types_list, number_of_dimensions=None):
    """Average consecutive blocks of patch rows into per-image rows.

    Parameters
    ----------
    data_matrix : numpy.ndarray or scipy.sparse matrix
        Matrix of per-patch values whose rows are grouped by image,
        ``number_of_patches`` consecutive rows per image.
    number_of_patches : int
        Number of patches per image.
    types_list : sequence
        Per-patch class labels.
    number_of_dimensions : int, optional
        Number of leading columns to retain. Defaults to all columns.

    Returns
    -------
    dict
        Dictionary with entries "image_weights" (per-image averaged
        matrix) and "image_types" (per-image class labels).
    """
    if sparse.issparse(data_matrix):
        data_matrix = np.asarray(data_matrix.todense())
    data_matrix = np.atleast_2d(np.asarray(data_matrix))
    if data_matrix.shape[0] == 1 and len(types_list) > 1:
        data_matrix = data_matrix.T
    if number_of_dimensions is not None:
        data_matrix = data_matrix[:, :number_of_dimensions]
    number_of_images = data_matrix.shape[0] // number_of_patches
    image_weights = data_matrix.reshape(number_of_images, number_of_patches, -1).mean(axis=1)
    image_types = np.asarray(types_list)[:: number_of_patches][:number_of_images]
    return {"image_weights": image_weights, "image_types": image_types}


def _class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    return {label: max_count / count for label, count in zip(unique, counts)}


def _accuracy(predicted, actual):
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    return float((predicted == actual).mean())


def _linear_svc(cost, class_weight, l1=False):
    from sklearn.svm import LinearSVC

    return LinearSVC(
        penalty="l1" if l1 else "l2",
        loss="squared_hinge",
        dual=False,
        C=cost,
        class_weight=class_weight,
    )


def _shuffled_folds(number_of_images, patches_per_image, number_of_folds, rng):
    shuffled_order = rng.permutation(number_of_images)
    image_folds = np.floor(np.arange(number_of_images) * number_of_folds / number_of_images).astype(int) + 1
    patch_indices = (
        shuffled_order[:, None] * patches_per_image + np.arange(patches_per_image)[None, :]
    ).ravel()
    patch_folds = np.repeat(image_folds, patches_per_image)
    return shuffled_order, image_folds, patch_indices, patch_folds


def compute_landscapes(
    parameters=None,
    number_of_cores=2,
    experiment_name=None,
    image_directories=None,
    patch_center_image_directories=None,
    directory_classes=None,
    data_results_directory=None,
    radius_of_patches=None,
    patch_ratio=2,
    pca=False,
    pca_components=1000,
    center_data=False,
    verbose=False,
    proportion=0.025,
    lower=0,
    upper=1,
    random_state=None,
):
    """Compute patch landscapes for directories of images.

    Featurizes every image in the given directories with
    :func:`patch_landscapes_from_image`, stacking the resulting
    landscape vectors into a single sparse matrix with class labels.
    Parameters may be supplied programmatically or via a parameters
    file; programmatic arguments take precedence over values from the
    file.

    Parameters
    ----------
    parameters : str, optional
        File path to csv file designating which folders to analyze. A
        template parameters.csv is included at
        https://github.com/P-Edwards/TDAExplore-ML. You can
        alternatively provide parameters via this function's arguments
        directly. If you provide both a parameters file and function
        arguments, the function arguments will overwrite the arguments
        from the csv file.
    number_of_cores : int
        The number of CPU cores for parallelization. Parallelization is
        over images, so allocating more cores than images provides no
        benefit.
    experiment_name : str, optional
        Prefix that will have additional information appended to form
        an informative file name for saving. Default is a random
        number.
    image_directories : sequence of str, optional
        Paths to directories holding image files to process.
    patch_center_image_directories : sequence, optional
        Sequence of the same length as ``image_directories`` with paths
        to directories holding image files. The files in
        ``patch_center_image_directories[i]`` will be matched, in
        alphabetical order, to the files in ``image_directories[i]``,
        and used for selecting the centers of patches. This is useful
        when e.g. ``patch_center_image_directories[i]`` contains masks
        for parts of the images in ``image_directories[i]`` to focus
        on. ``patch_center_image_directories[i]`` may also be set to
        None instead of a path, in which case no masked images will be
        used for center selection with ``image_directories[i]``.
    directory_classes : sequence of str, optional
        Sequence of same length as image directories.
        ``directory_classes[i]`` specifies the class of the images in
        ``image_directories[i]``.
    data_results_directory : str, optional
        File path (relative to current working directory) where data
        results can be saved. This function will only produce a file
        path for saving, the user must save the results themselves if
        desired. ./tda_explore_results is the default.
    radius_of_patches : int, optional
        Pixel radius of patches. Default is 2.5% of image length.
    patch_ratio : float
        The number of patches sampled per image will be
        patch_ratio*(PIXEL AREA OF IMAGE)/(PIXEL AREA OF SINGLE PATCH).
        Default is 2.
    pca : bool
        If set to True, transforms landscapes after computation by
        projecting onto leading principal components, then scaling.
        Default is False.
    pca_components : int
        Number of principal components used when ``pca`` is True.
        Default is 1000.
    center_data : bool
        If set to True, centers the columns of the landscape matrix.
        Default is False.
    verbose : bool
        If set to True, outputs some progress information using print.
        Default is False.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for random sampling. Default is None, in
        which case fresh entropy is used.

    Returns
    -------
    dict
        Dictionary ml_results whose entries contain the computation
        results, including "summaries" (matrix of computed landscapes
        for patches), "patch_types" (class of image each landscape
        comes from), "class_names", "patches_per_image",
        "image_file_names", "radius", "data_results_directory", and
        "data_name_stem".
    """
    rng = np.random.default_rng(random_state)
    file_parameters = read_parameters(parameters) if parameters is not None else {}
    ml_results = {}

    if image_directories is not None:
        image_type_names = list(image_directories)
    elif "image_directories" in file_parameters:
        image_type_names = file_parameters["image_directories"]
    else:
        raise ValueError("Image directories not specified, and there is no default.")

    if directory_classes is not None:
        class_labels = list(directory_classes)
    elif "directory_classes" in file_parameters:
        class_labels = file_parameters["directory_classes"]
    else:
        warnings.warn("No classes for directory specified, using directory names as a default.")
        class_labels = list(image_type_names)
    class_names = list(dict.fromkeys(class_labels))

    if patch_center_image_directories is not None:
        patch_center_names = list(patch_center_image_directories)
    elif "patch_center_images_directories" in file_parameters:
        patch_center_names = file_parameters["patch_center_images_directories"]
    else:
        patch_center_names = [None] * len(image_type_names)

    image_file_names_by_directory = []
    patch_center_file_names_by_directory = []
    for i, directory in enumerate(image_type_names):
        files = sorted(os.path.join(directory, name) for name in os.listdir(directory))
        image_file_names_by_directory.append(files)
        center_directory = patch_center_names[i] if i < len(patch_center_names) else None
        if center_directory in (None, False, "FALSE", ""):
            patch_center_file_names_by_directory.append([None] * len(files))
        else:
            center_files = sorted(
                os.path.join(center_directory, name) for name in os.listdir(center_directory)
            )
            patch_center_file_names_by_directory.append(center_files)

    if radius_of_patches is None:
        if "radius_of_patches" in file_parameters:
            radius_of_patches = int(file_parameters["radius_of_patches"][0])
        else:
            warnings.warn("No pixel radius for patches specified. Using default, which is 2.5% of image length.")
            first_image = read_image(image_file_names_by_directory[0][0])
            radius_of_patches = math.floor(0.025 * first_image.shape[0])
    radius_of_patches = int(radius_of_patches)

    image_for_dimensions = read_image(image_file_names_by_directory[0][0])
    pixel_area_of_images = image_for_dimensions.shape[0] * image_for_dimensions.shape[1]
    patches_per_image = max(
        math.floor(patch_ratio * pixel_area_of_images / (math.pi * radius_of_patches**2)), 2
    )

    if experiment_name is None:
        if "experiment_name" in file_parameters:
            experiment_name = str(file_parameters["experiment_name"][0])
        else:
            experiment_name = str(math.floor(rng.uniform(0, 1e6)))

    if data_results_directory is None:
        if "data_results_directory" in file_parameters:
            data_results_directory = str(file_parameters["data_results_directory"][0])
        else:
            data_results_directory = "tda_explore_results"
    if not os.path.isdir(data_results_directory):
        warnings.warn("Results directory didn't exist, creating " + data_results_directory)
        os.makedirs(data_results_directory, exist_ok=True)
    ml_results["data_results_directory"] = data_results_directory
    current_time = time.strftime("%b-%d-%M")
    data_name_stem = f"{experiment_name}patches_{patch_ratio}_radius_{radius_of_patches}_{current_time}"
    if upper - lower < 1:
        data_name_stem = data_name_stem + "_radialthresh"
    ml_results["data_name_stem"] = data_name_stem

    image_file_names = [name for files in image_file_names_by_directory for name in files]
    patch_center_file_names = [name for files in patch_center_file_names_by_directory for name in files]
    type_vector = np.repeat(
        [
            class_labels[i]
            for i, files in enumerate(image_file_names_by_directory)
            for _ in files
        ],
        patches_per_image,
    )

    seeds = rng.spawn(len(image_file_names)) if random_state is not None else [None] * len(image_file_names)

    def featurize(image_name, center_name, seed):
        if verbose:
            print("Started image", image_name)
        result = patch_landscapes_from_image(
            image_name,
            patches_per_image,
            pixel_radius_for_patches=radius_of_patches,
            patch_center_image=center_name,
            proportion_of_patch_sparse=proportion,
            lower_threshold=lower,
            upper_threshold=upper,
            random_state=seed,
        )
        if verbose:
            print("Finished image", image_name)
        return result

    if number_of_cores is not None and number_of_cores != 1:
        from joblib import Parallel, delayed

        blocks = Parallel(n_jobs=number_of_cores)(
            delayed(featurize)(image_name, center_name, seed)
            for image_name, center_name, seed in zip(image_file_names, patch_center_file_names, seeds)
        )
    else:
        blocks = [
            featurize(image_name, center_name, seed)
            for image_name, center_name, seed in zip(image_file_names, patch_center_file_names, seeds)
        ]
    data = sparse.vstack(blocks, format="csr")

    if center_data:
        data = np.asarray(data.todense())
        data = data - data.mean(axis=0)

    if pca:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler

        n_components = min(pca_components, min(data.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=rng.integers(2**32))
        data = svd.fit_transform(data)
        data = StandardScaler().fit_transform(data)
        ml_results["landscapes_svd"] = svd

    ml_results["summaries"] = data
    ml_results["patch_types"] = type_vector
    ml_results["class_names"] = class_names
    ml_results["patches_per_image"] = patches_per_image
    ml_results["patch_ratio"] = patch_ratio
    ml_results["image_file_names"] = image_file_names
    ml_results["radius"] = radius_of_patches
    return ml_results


def run_svm(
    parameters=None,
    number_of_folds=5,
    verbose=False,
    landscapes=None,
    random_state=None,
    **landscape_kwargs,
):
    """Train and cross-validate a linear SVM classifier from image landscapes.

    Computes patch landscapes (or reuses precomputed ones), then
    performs image-grouped cross-validation of a two-class linear SVM.
    Each fold trains a per-patch SVM and a second-stage SVM which
    classifies images from the average of their patch scores.

    Parameters
    ----------
    parameters : str, optional
        File path to csv file designating which folders to analyze. See
        :func:`compute_landscapes`.
    number_of_folds : int
        The number of folds to use in SVM cross-validation. The default
        is 5.
    verbose : bool
        If set to True, outputs some progress information using print.
        Default is False.
    landscapes : dict, optional
        Precomputed results from :func:`compute_landscapes`. When
        supplied, featurization is skipped.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for random sampling. Default is None, in
        which case fresh entropy is used.
    **landscape_kwargs
        Additional arguments passed to :func:`compute_landscapes`.

    Returns
    -------
    dict
        Dictionary ml_results whose entries contain the computation
        results. In addition to the entries of
        :func:`compute_landscapes`, "svm" holds a list containing
        results from each fold of SVM training-testing computation.
    """
    rng = np.random.default_rng(random_state)
    use_pca = landscape_kwargs.get("pca", False)
    if landscapes is None:
        ml_results = compute_landscapes(
            parameters=parameters, verbose=verbose, random_state=rng, **landscape_kwargs
        )
    else:
        ml_results = dict(landscapes)

    if verbose:
        print("Starting per-landscape SVM")

    data = ml_results["summaries"]
    type_vector = np.asarray(ml_results["patch_types"])
    patches_per_image = ml_results["patches_per_image"]
    image_file_names = np.asarray(ml_results["image_file_names"])
    number_of_images = len(image_file_names)
    l1 = bool(use_pca)

    shuffled_order, image_folds, patch_indices, patch_folds = _shuffled_folds(
        number_of_images, patches_per_image, number_of_folds, rng
    )
    shuffled_data = data[patch_indices]
    shuffled_types = type_vector[patch_indices]

    cost = heuristic_C(shuffled_data[patch_folds == 1])

    patch_accuracies = np.zeros(number_of_folds)
    image_accuracies = np.zeros(number_of_folds)
    ml_results["svm"] = []

    for fold in range(1, number_of_folds + 1):
        if verbose:
            print("Starting cross validation fold number", fold)
        test_mask = patch_folds == fold
        train_mask = ~test_mask

        svm_model = _linear_svc(cost, _class_weights(shuffled_types[train_mask]), l1=l1)
        svm_model.fit(shuffled_data[train_mask], shuffled_types[train_mask])

        fold_results = {
            "testing_data": shuffled_data[test_mask],
            "testing_labels": shuffled_types[test_mask],
        }

        predictions = svm_model.predict(shuffled_data[test_mask])
        patch_accuracies[fold - 1] = _accuracy(predictions, shuffled_types[test_mask])

        # Classification with average transformed image data:
        # (1) Transform all patch landscapes into single numbers using model
        # (2) Average all numbers across each image
        # (3) Train and test an SVM model for classifying images
        transformed = shuffled_data @ svm_model.coef_.ravel() + svm_model.intercept_[0]
        averaged = average_vectors_for_images(transformed, patches_per_image, shuffled_types)
        image_data = averaged["image_weights"]
        image_types = averaged["image_types"]

        image_test_mask = image_folds == fold
        image_train_mask = ~image_test_mask
        image_cost = heuristic_C(image_data[image_folds == 1])

        image_svm_model = _linear_svc(image_cost, _class_weights(image_types[image_train_mask]), l1=l1)
        image_svm_model.fit(image_data[image_train_mask], image_types[image_train_mask])
        image_predictions = image_svm_model.predict(image_data[image_test_mask])
        image_accuracies[fold - 1] = _accuracy(image_predictions, image_types[image_test_mask])

        fold_results["svm_model"] = svm_model
        fold_results["patch_svm_model"] = svm_model
        fold_results["image_svm_model"] = image_svm_model
        fold_results["svm_image_training_indices"] = shuffled_order[image_train_mask]
        fold_results["svm_patch_accuracies"] = patch_accuracies
        fold_results["svm_image_accuracies"] = image_accuracies
        fold_results["svm_cost"] = cost
        ml_results["svm"].append(fold_results)

    if verbose:
        print("Radius")
        print(ml_results["radius"])
        print("Average patch accuracies")
        print(patch_accuracies.mean())
        print("Average image accuracies")
        print(image_accuracies.mean())
    return ml_results


def run_multisvm(
    parameters=None,
    number_of_folds=5,
    verbose=False,
    landscapes=None,
    random_state=None,
    **landscape_kwargs,
):
    """Train and cross-validate a multi-class linear SVM from image landscapes.

    Computes patch landscapes (or reuses precomputed ones), then
    performs image-grouped cross-validation of a multi-class linear
    SVM. Each fold trains a per-patch SVM; images are classified by a
    second-stage SVM trained on per-image averages of the per-patch
    decision values.

    Parameters
    ----------
    parameters : str, optional
        File path to csv file designating which folders to analyze. See
        :func:`compute_landscapes`.
    number_of_folds : int
        The number of folds to use in SVM cross-validation. The default
        is 5.
    verbose : bool
        If set to True, outputs some progress information using print.
        Default is False.
    landscapes : dict, optional
        Precomputed results from :func:`compute_landscapes`. When
        supplied, featurization is skipped.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for random sampling. Default is None, in
        which case fresh entropy is used.
    **landscape_kwargs
        Additional arguments passed to :func:`compute_landscapes`.
        Landscape columns are centered by default for this pipeline.

    Returns
    -------
    dict
        Dictionary ml_results whose entries contain the computation
        results. In addition to the entries of
        :func:`compute_landscapes`, "multisvm" holds a list containing
        results from each fold of SVM training-testing computation.
    """
    rng = np.random.default_rng(random_state)
    if landscapes is None:
        landscape_kwargs.setdefault("center_data", True)
        ml_results = compute_landscapes(
            parameters=parameters, verbose=verbose, random_state=rng, **landscape_kwargs
        )
    else:
        ml_results = dict(landscapes)

    if verbose:
        print("Starting per-landscape SVM")

    data = ml_results["summaries"]
    type_vector = np.asarray(ml_results["patch_types"])
    patches_per_image = ml_results["patches_per_image"]
    number_of_images = len(ml_results["image_file_names"])

    shuffled_order, image_folds, patch_indices, patch_folds = _shuffled_folds(
        number_of_images, patches_per_image, number_of_folds, rng
    )
    shuffled_data = data[patch_indices]
    shuffled_types = type_vector[patch_indices]

    cost = heuristic_C(shuffled_data[patch_folds == 1])

    patch_accuracies = np.zeros(number_of_folds)
    image_accuracies = np.zeros(number_of_folds)
    ml_results["multisvm"] = []

    for fold in range(1, number_of_folds + 1):
        if verbose:
            print("Starting cross validation fold number", fold)
        test_mask = patch_folds == fold
        train_mask = ~test_mask

        svm_model = _linear_svc(cost, _class_weights(shuffled_types[train_mask]))
        svm_model.fit(shuffled_data[train_mask], shuffled_types[train_mask])

        fold_results = {
            "testing_data": shuffled_data[test_mask],
            "testing_labels": shuffled_types[test_mask],
        }

        predictions = svm_model.predict(shuffled_data[test_mask])
        patch_accuracies[fold - 1] = _accuracy(predictions, shuffled_types[test_mask])

        # Classification with average decision value vectors:
        # Predict decision values for each patch, then take an average
        # over all patches in each image.
        decision_values = svm_model.decision_function(shuffled_data)
        if decision_values.ndim == 1:
            decision_values = np.column_stack([decision_values, -decision_values])

        averaged = average_vectors_for_images(decision_values, patches_per_image, shuffled_types)
        image_data = averaged["image_weights"]
        image_types = averaged["image_types"]

        image_test_mask = image_folds == fold
        image_train_mask = ~image_test_mask
        image_cost = heuristic_C(image_data[image_folds == 1])

        image_svm_model = _linear_svc(image_cost, _class_weights(image_types[image_train_mask]), l1=True)
        image_svm_model.fit(image_data[image_train_mask], image_types[image_train_mask])
        image_predictions = image_svm_model.predict(image_data[image_test_mask])
        image_accuracies[fold - 1] = _accuracy(image_predictions, image_types[image_test_mask])

        fold_results["svm_model"] = svm_model
        fold_results["patch_svm_model"] = svm_model
        fold_results["image_svm_model"] = image_svm_model
        fold_results["svm_image_training_indices"] = shuffled_order[image_train_mask]
        fold_results["svm_patch_accuracies"] = patch_accuracies
        fold_results["svm_image_accuracies"] = image_accuracies
        ml_results["multisvm"].append(fold_results)

    if verbose:
        print("Radius")
        print(ml_results["radius"])
        print("Average patch accuracies")
        print(patch_accuracies.mean())
        print("Average image accuracies")
        print(image_accuracies.mean())
    return ml_results


def run_random_forest(
    parameters=None,
    number_of_folds=5,
    verbose=False,
    landscapes=None,
    random_state=None,
    **landscape_kwargs,
):
    """Train and cross-validate a random forest from image landscapes.

    Computes patch landscapes (or reuses precomputed ones), then
    performs image-grouped cross-validation of a random forest
    classifier. Supports multiclass. Could take prohibitively long if
    pca is not True. Images are classified by averaging predicted class
    probabilities over each image's patches; the highest average
    probability wins the image.

    Parameters
    ----------
    parameters : str, optional
        File path to csv file designating which folders to analyze. See
        :func:`compute_landscapes`.
    number_of_folds : int
        The number of folds to use in cross-validation. The default is
        5.
    verbose : bool
        If set to True, outputs some progress information using print.
        Default is False.
    landscapes : dict, optional
        Precomputed results from :func:`compute_landscapes`. When
        supplied, featurization is skipped.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for random sampling. Default is None, in
        which case fresh entropy is used.
    **landscape_kwargs
        Additional arguments passed to :func:`compute_landscapes`.

    Returns
    -------
    dict
        Dictionary ml_results whose entries contain the computation
        results. In addition to the entries of
        :func:`compute_landscapes`, "rand_forest" holds a list
        containing results from each fold of training-testing
        computation.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(random_state)
    if landscapes is None:
        ml_results = compute_landscapes(
            parameters=parameters, verbose=verbose, random_state=rng, **landscape_kwargs
        )
    else:
        ml_results = dict(landscapes)

    if verbose:
        print("Starting per-landscape random forest")

    data = ml_results["summaries"]
    type_vector = np.asarray(ml_results["patch_types"])
    patches_per_image = ml_results["patches_per_image"]
    number_of_images = len(ml_results["image_file_names"])

    shuffled_order, image_folds, patch_indices, patch_folds = _shuffled_folds(
        number_of_images, patches_per_image, number_of_folds, rng
    )
    shuffled_data = data[patch_indices]
    shuffled_types = type_vector[patch_indices]

    patch_accuracies = np.zeros(number_of_folds)
    image_accuracies = np.zeros(number_of_folds)
    ml_results["rand_forest"] = []

    for fold in range(1, number_of_folds + 1):
        if verbose:
            print("Starting random forest cross validation fold number", fold)
        test_mask = patch_folds == fold
        train_mask = ~test_mask

        rf_model = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", random_state=rng.integers(2**32)
        )
        rf_model.fit(shuffled_data[train_mask], shuffled_types[train_mask])

        fold_results = {
            "testing_data": shuffled_data[test_mask],
            "testing_labels": shuffled_types[test_mask],
        }

        predictions = rf_model.predict(shuffled_data[test_mask])
        patch_accuracies[fold - 1] = _accuracy(predictions, shuffled_types[test_mask])

        # Classification with average probability vectors:
        # Predict class probabilities for each testing patch, then take
        # an average over all patches in each testing image. Highest
        # average probability wins the image.
        probabilities = rf_model.predict_proba(shuffled_data[test_mask])
        averaged = average_vectors_for_images(probabilities, patches_per_image, shuffled_types[test_mask])
        image_predictions = rf_model.classes_[np.argmax(averaged["image_weights"], axis=1)]
        image_accuracies[fold - 1] = _accuracy(image_predictions, averaged["image_types"])

        fold_results["image_predictions"] = image_predictions
        fold_results["ground_values"] = averaged["image_types"]
        fold_results["rf_model"] = rf_model
        fold_results["rf_image_training_indices"] = shuffled_order[image_folds != fold]
        fold_results["rf_patch_accuracies"] = patch_accuracies
        fold_results["rf_image_accuracies"] = image_accuracies
        ml_results["rand_forest"].append(fold_results)

    if verbose:
        print("Radius")
        print(ml_results["radius"])
        print("Average RF patch accuracies")
        print(patch_accuracies.mean())
        print("Average RF image accuracies")
        print(image_accuracies.mean())
    return ml_results
