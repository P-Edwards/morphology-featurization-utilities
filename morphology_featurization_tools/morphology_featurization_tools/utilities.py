import numpy as np 

def permutation_test_mean_diff(X, y, n_permutations=1000, random_state=42):
    """
    Permutation test using the 2-norm of the difference in class means
    as the test statistic.

    Args:
        X:               Feature matrix (n_samples, n_features)
        y:               Class labels (n_samples,)
        n_permutations:  Number of permutations
        random_state:    RNG seed

    Returns:
        observed_stat: test statistic on real labels
        null_stats:    array of test statistics under permuted labels
        p_value:       fraction of null stats >= observed stat
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    def test_statistic(X, y):
        means = [X[y == c].mean(axis=0) for c in classes]
        # Sum of pairwise 2-norms for >2 classes
        total = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                total += np.linalg.norm(means[i] - means[j])
        return total

    observed_stat = test_statistic(X, y)

    null_stats = np.array([
        test_statistic(X, rng.permutation(y))
        for _ in range(n_permutations)
    ])

    p_value = np.mean(null_stats >= observed_stat)

    print(f"Observed statistic: {observed_stat:.4f}")
    print(f"Null mean:          {null_stats.mean():.4f} ± {null_stats.std():.4f}")
    print(f"p-value:            {p_value:.4f}")

    return observed_stat, null_stats, p_value

def permutation_test_mean_diff_pipeline(
    pipeline,
    X,
    y,
    n_permutations=1000,
    random_state=42,
    **fit_params,
):
    """
    Permutation test using the 2-norm of class mean differences,
    applied to features extracted by a sklearn pipeline.

    The pipeline is fit on the real data once to extract features,
    then refit on each permutation of labels if the pipeline contains
    fitted steps. If the pipeline is fully deterministic (no fitted
    steps that depend on y), features are precomputed once.

    Args:
        pipeline:        sklearn Pipeline ending before a classifier
        X:               Raw input array
        y:               Class labels
        n_permutations:  Number of permutations
        random_state:    RNG seed
        **fit_params:    Additional params passed to pipeline.fit_transform
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    def test_statistic(X_feat, y):
        means = [X_feat[y == c].mean(axis=0) for c in classes]
        total = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                total += np.linalg.norm(means[i] - means[j])
        return total

    # Precompute features once since pipeline doesn't depend on y
    X_feat = pipeline.fit_transform(X, **fit_params)

    observed_stat = test_statistic(X_feat, y)

    null_stats = np.array([
        test_statistic(X_feat, rng.permutation(y))
        for _ in range(n_permutations)
    ])

    p_value = np.mean(null_stats >= observed_stat)

    print(f"Observed statistic: {observed_stat:.4f}")
    print(f"Null mean:          {null_stats.mean():.4f} ± {null_stats.std():.4f}")
    print(f"p-value:            {p_value:.4f}")

    return observed_stat, null_stats, p_value