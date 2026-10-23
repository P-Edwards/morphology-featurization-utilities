"""
Batch-effect normalizers for well-level profiles, fitted on negative controls.
Both have the interface

    fit(X, control_mask=..., batch_labels=...)
    transform(X, batch_labels=...)

  TypicalVariationNormalization  PCA on all controls, then a full covariance
                                 whitening and recolouring per batch. The k x k
                                 covariance of each batch is estimated from that
                                 batch's controls, which limits the usable rank.
  MADRobustize                   per batch and feature, subtract the control
                                 median and divide by the control MAD
                                 (pycytominer's mad_robustize, as in the JUMP
                                 profiling recipe). Features keep their meaning.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

__all__ = ["TypicalVariationNormalization", "MADRobustize"]


class TypicalVariationNormalization(BaseEstimator, TransformerMixin):
    """
    Typical Variation Normalization (TVN) with per-batch correlation alignment.

    Fit:
      - PCA on all negative controls (variance whitening basis)
      - C_all: covariance of all controls in PCA space
      - C_i:   per-batch covariance of controls in PCA space

    Transform for sample from batch i:
      1. Project into PCA space (removes dominant variance)
      2. Whiten with C_i     (removes per-batch correlations)
      3. Recolour with C_all (restores shared correlation structure)

    Args:
        n_components: PCA components to retain. If None, keeps all.
        epsilon:      Regularisation added to eigenvalues for matrix stability.
    """

    def __init__(self, n_components: int | None = None, epsilon: float = 1e-6):
        self.n_components = n_components
        self.epsilon = epsilon

    def _cov_sqrt_inv(self, C: np.ndarray) -> np.ndarray:
        """Compute C^{-1/2} via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon)) @ eigenvectors.T

    def _cov_sqrt(self, C: np.ndarray) -> np.ndarray:
        """Compute C^{1/2} via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues + self.epsilon)) @ eigenvectors.T

    def fit(self, X: np.ndarray, y=None,
            control_mask: np.ndarray | None = None,
            batch_labels: np.ndarray | None = None):

        if control_mask is None:
            raise ValueError("control_mask must be provided.")
        if batch_labels is None:
            raise ValueError("batch_labels must be provided.")

        controls = X[control_mask]
        control_batches = batch_labels[control_mask]

        if len(controls) == 0:
            raise ValueError("No control samples found in this fold.")

        # PCA on all controls, whitened
        self.pca_ = PCA(n_components=self.n_components, whiten=True)
        controls_pca = self.pca_.fit_transform(controls)

        # covariance of all controls in PCA space
        self.C_all_sqrt_ = self._cov_sqrt(np.cov(controls_pca, rowvar=False))

        # covariance of each batch's controls in PCA space
        self.C_i_sqrt_inv_ = {}
        for batch in np.unique(control_batches):
            batch_controls_pca = controls_pca[control_batches == batch]
            C_i = np.cov(batch_controls_pca, rowvar=False)
            self.C_i_sqrt_inv_[batch] = self._cov_sqrt_inv(C_i)

        self.unique_batches_ = np.unique(control_batches)
        return self

    def transform(self, X: np.ndarray,
                  batch_labels: np.ndarray | None = None) -> np.ndarray:
        if batch_labels is None:
            raise ValueError("batch_labels must be provided for transform.")

        X_pca = self.pca_.transform(X)

        result = np.empty_like(X_pca)
        for batch in np.unique(batch_labels):
            mask = batch_labels == batch
            if batch not in self.C_i_sqrt_inv_:
                raise ValueError(
                    f"Batch '{batch}' was not seen during fit. "
                    f"Known batches: {list(self.C_i_sqrt_inv_.keys())}"
                )
            # whiten with C_i^{-1/2}, recolour with C_all^{1/2}
            result[mask] = X_pca[mask] @ self.C_i_sqrt_inv_[batch].T @ self.C_all_sqrt_.T

        return result

    def predict(self, X: np.ndarray, batch_labels: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X, batch_labels=batch_labels)    

    def set_fit_request(self, *, control_mask=True, batch_labels=True):
        return super().set_fit_request(control_mask=control_mask, batch_labels=batch_labels)

    def set_transform_request(self, *, batch_labels=True):
        return super().set_transform_request(batch_labels=batch_labels)


class MADRobustize(BaseEstimator, TransformerMixin):
    """
    Robust standardization against negative controls, per batch and feature:

        x' = (x - median(control x)) / (MAD(control x) * scale + epsilon)

    the normalization step of the JUMP profiling recipe (pycytominer's
    mad_robustize with samples='negcon'). Batches without controls, at fit or
    at transform, use the pooled control median and MAD.

    Args:
        scale:    1.4826 makes the MAD estimate the standard deviation of
                  normal data; 1.0 gives the raw MAD.
        epsilon:  added to the scaled MAD before dividing.
        clip:     symmetric clip after scaling, so a feature whose control MAD
                  is near zero cannot dominate distances. None disables it.
    """

    def __init__(self, scale: float = 1.4826, epsilon: float = 1e-18,
                 clip: float | None = 50.0):
        self.scale = scale
        self.epsilon = epsilon
        self.clip = clip

    def fit(self, X: np.ndarray, y=None,
            control_mask: np.ndarray | None = None,
            batch_labels: np.ndarray | None = None):
        if control_mask is None:
            raise ValueError("control_mask must be provided.")
        if batch_labels is None:
            raise ValueError("batch_labels must be provided.")

        X = np.asarray(X, dtype=float)
        control_mask = np.asarray(control_mask, dtype=bool)
        batch_labels = np.asarray(batch_labels)

        if not control_mask.any():
            raise ValueError("No control samples found.")

        self.center_ = {}
        self.scale_ = {}
        self.batches_without_controls_ = []

        for batch in np.unique(batch_labels):
            in_batch = (batch_labels == batch) & control_mask
            if not in_batch.any():
                # Such batches use the pooled estimate at transform.
                self.batches_without_controls_.append(batch)
                continue
            controls = X[in_batch]
            with np.errstate(invalid="ignore"):
                median = np.nanmedian(controls, axis=0)
                mad = np.nanmedian(np.abs(controls - median), axis=0)
            median = np.nan_to_num(median, nan=0.0)
            mad = np.nan_to_num(mad, nan=0.0)
            self.center_[batch] = median
            self.scale_[batch] = mad * self.scale + self.epsilon

        if not self.center_:
            raise ValueError("No batch had any control samples.")

        # Pooled estimate for batches without controls.
        with np.errstate(invalid="ignore"):
            pooled_median = np.nanmedian(X[control_mask], axis=0)
            pooled_mad = np.nanmedian(np.abs(X[control_mask] - pooled_median), axis=0)
        self.pooled_center_ = np.nan_to_num(pooled_median, nan=0.0)
        self.pooled_scale_ = np.nan_to_num(pooled_mad, nan=0.0) * self.scale + self.epsilon
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray,
                  batch_labels: np.ndarray | None = None) -> np.ndarray:
        if batch_labels is None:
            raise ValueError("batch_labels must be provided for transform.")

        X = np.asarray(X, dtype=float)
        batch_labels = np.asarray(batch_labels)
        result = np.empty_like(X)

        for batch in np.unique(batch_labels):
            mask = batch_labels == batch
            center = self.center_.get(batch, self.pooled_center_)
            scale = self.scale_.get(batch, self.pooled_scale_)
            result[mask] = (X[mask] - center) / scale

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip is not None:
            np.clip(result, -self.clip, self.clip, out=result)
        return result

    def predict(self, X: np.ndarray, batch_labels: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X, batch_labels=batch_labels)

    def set_fit_request(self, *, control_mask=True, batch_labels=True):
        return super().set_fit_request(control_mask=control_mask,
                                       batch_labels=batch_labels)

    def set_transform_request(self, *, batch_labels=True):
        return super().set_transform_request(batch_labels=batch_labels)
