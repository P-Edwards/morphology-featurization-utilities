"""
Retrospective multi-image illumination correction (Caicedo et al. 2017;
Bray et al. 2016).

For each group of fields sharing an optical path (here, a plate) and each
channel, the mean field is smoothed with a wide Gaussian and scaled to unit
mean; each field is divided by its group's function, which removes the shape
of the shading but not the overall intensity. fit keeps only running sums, so
memory does not grow with the number of fields.

Used for the engineered features; the OpenPhenom autoencoder receives
uncorrected fields.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class IlluminationCorrection:
    """
    Per-group, per-channel retrospective illumination correction.

    Parameters
    ----------
    smoothing_sigma : float
        Standard deviation (pixels) of the Gaussian used to smooth each
        group's average field into an illumination function. Should be large
        relative to cell size so the function captures only the slow shading,
        not cellular structure. Default 100.
    epsilon : float
        Floor added to the illumination function before division to avoid
        divide-by-zero. Default 1e-6.

    Attributes (set after fit)
    ----------
    illumination_functions_ : dict
        Maps group key -> np.ndarray illumination function. For multichannel
        data the function has shape (C, H, W); for single-channel, (H, W).
        Each is normalised to unit mean.
    """

    def __init__(self, smoothing_sigma: float = 100.0, epsilon: float = 1e-6):
        self.smoothing_sigma = smoothing_sigma
        self.epsilon = epsilon

    def _smooth(self, mean_field: np.ndarray) -> np.ndarray:
        """Smooth a (H, W) or (C, H, W) mean field into an illumination function."""
        if mean_field.ndim == 2:
            smoothed = ndimage.gaussian_filter(mean_field, self.smoothing_sigma)
        else:
            smoothed = np.stack([
                ndimage.gaussian_filter(mean_field[c], self.smoothing_sigma)
                for c in range(mean_field.shape[0])
            ])
        mean_value = smoothed.mean()
        if mean_value > 0:
            smoothed = smoothed / mean_value
        return smoothed

    def fit(
        self,
        fields: list[np.ndarray],
        group_labels: np.ndarray | list,
    ) -> "IlluminationCorrection":
        """
        Estimate an illumination function per group from its fields.

        Parameters
        ----------
        fields : list of np.ndarray
            Field images, each (H, W) or (C, H, W). All fields in a group
            must share spatial shape.
        group_labels : array-like
            Group identifier per field (e.g. plate id), aligned with `fields`.

        Returns
        -------
        self
        """
        group_labels = np.asarray(group_labels, dtype=object)

        running_sum: dict = {}
        running_count: dict = {}

        for field_image, group in zip(fields, group_labels):
            field_float = field_image.astype(np.float64)
            if group not in running_sum:
                running_sum[group] = field_float.copy()
                running_count[group] = 1
            else:
                running_sum[group] += field_float
                running_count[group] += 1

        self.illumination_functions_ = {
            group: self._smooth(running_sum[group] / running_count[group])
            for group in running_sum
        }
        return self

    def transform(
        self,
        fields: list[np.ndarray],
        group_labels: np.ndarray | list,
    ) -> list[np.ndarray]:
        """
        Divide each field by its group's illumination function.

        Parameters
        ----------
        fields : list of np.ndarray
            Field images to correct, aligned with `group_labels`.
        group_labels : array-like
            Group identifier per field.

        Returns
        -------
        list of np.ndarray
            Corrected fields, same shapes and dtypes as the inputs (values
            are recomputed in float32 then cast back to the input dtype).

        Notes
        -----
        The division is done in float32, which is ample for a smooth gain on
        16-bit data. float16 would overflow above 65504.
        """
        if not hasattr(self, "illumination_functions_"):
            raise RuntimeError("IlluminationCorrection must be fit before transform.")

        group_labels = np.asarray(group_labels, dtype=object)
        corrected = []

        for field_image, group in zip(fields, group_labels):
            if group not in self.illumination_functions_:
                raise ValueError(
                    f"Group '{group}' was not seen during fit. "
                    f"Known groups: {list(self.illumination_functions_)}"
                )
            function = self.illumination_functions_[group]
            result = field_image.astype(np.float32) / (function + self.epsilon).astype(np.float32)

            # Round (astype would truncate) and clip to the integer range.
            if np.issubdtype(field_image.dtype, np.integer):
                info = np.iinfo(field_image.dtype)
                result = np.clip(np.rint(result), info.min, info.max)
            corrected.append(result.astype(field_image.dtype))

        return corrected

    def fit_transform(
        self,
        fields: list[np.ndarray],
        group_labels: np.ndarray | list,
    ) -> list[np.ndarray]:
        return self.fit(fields, group_labels).transform(fields, group_labels)
