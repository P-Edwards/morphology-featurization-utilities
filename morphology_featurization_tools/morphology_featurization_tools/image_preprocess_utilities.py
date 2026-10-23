import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable
from skimage import img_as_float
from skimage.restoration import denoise_bilateral
from functools import partial



class ImagePreprocessor(TransformerMixin, BaseEstimator):
    """
    Sklearn-compatible preprocessor for batches of grayscale or multi-channel images.

    Args:
        per_image_preprocessor: Applied to each 2D slice (per channel, per sample).
                                 Mutually exclusive with per_tensor_preprocessor.
        per_tensor_preprocessor: Applied to each 3D slice X[index, :, :, :].
                                  Mutually exclusive with per_image_preprocessor.
    """

    def __init__(
        self,
        per_image_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
        per_tensor_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if (per_image_preprocessor is None) == (per_tensor_preprocessor is None):
            raise ValueError("Provide exactly one of per_image_preprocessor or per_tensor_preprocessor.")

        self.per_image_preprocessor = per_image_preprocessor
        self.per_tensor_preprocessor = per_tensor_preprocessor

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.stack([self.per_image_preprocessor(X[i]) for i in range(len(X))])
        elif X.ndim == 4:
            if self.per_tensor_preprocessor is not None:
                return np.stack([self.per_tensor_preprocessor(X[i]) for i in range(len(X))])
            else:
                return np.stack([
                    np.stack([self.per_image_preprocessor(X[i, c]) for c in range(X.shape[1])])
                    for i in range(len(X))
                ])
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {X.shape}.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


class DenoiseBilateralPreprocessor(ImagePreprocessor):
    """
    Applies skimage's denoise_bilateral to each 2D image slice.

    Parameters mirror denoise_bilateral's own defaults.
    """

    def __init__(
        self,
        win_size: int | None = None,
        sigma_color: float | None = None,
        sigma_spatial: float | None = 1,
        bins: int = 10000,
        mode: str = 'constant',
        cval: float = 0,
    ):
        self.win_size = win_size
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
        self.bins = bins
        self.mode = mode
        self.cval = cval

        super().__init__(
            per_image_preprocessor=partial(
                denoise_bilateral,
                win_size=win_size,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                bins=bins,
                mode=mode,
                cval=cval,
            )
        )


def _full_reducer(image, lower_percentile, upper_percentile):
        lower_thresh = lower_percentile
        upper_thresh = upper_percentile

        lower_mask = image < lower_thresh
        upper_mask = image > upper_thresh
        
        preprocessed = image.copy()
        
        # Bottom percentile: set to 0 (black/background)
        preprocessed[lower_mask] = 0
        
        # Top percentile: set to max value (saturate bright outliers)
        max_value = upper_thresh
        preprocessed[upper_mask] = max_value

        return preprocessed

class ReduceOutlyingIntensities(ImagePreprocessor):
    """
    Clip outlying intensities of each image: values below `lower_percentile`
    are set to 0 and values above `upper_percentile` are set to
    `upper_percentile`.

    Both arguments are absolute intensity thresholds, not percentiles; on
    images scaled to [0, 1] (e.g. after FloatConverter) they are fractions of
    the full intensity range.
    """

    def __init__(
        self,
        lower_percentile: float | None = 5.0/100,
        upper_percentile: float | None = 97.0/100
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        super().__init__(
            per_image_preprocessor = partial(_full_reducer,
                                             lower_percentile = self.lower_percentile,
                                             upper_percentile = self.upper_percentile
                                             )
        )


class ChannelSelector(ImagePreprocessor):
    """
    Select a single channel from a batch of multi-channel images.

    Input:
    - 4D (N, C, H, W)

    Output:
    - 3D (N, H, W)
    """

    def __init__(self, channel: int = 0):
        self.channel = channel
        super().__init__(per_tensor_preprocessor=lambda image: image[self.channel])

# Converts each image to floating point (skimage.img_as_float).
FloatConverter = ImagePreprocessor(per_image_preprocessor=img_as_float)
# Divides each image by its maximum.
RescaleToZeroOne = ImagePreprocessor(per_image_preprocessor=lambda image: image/np.max(image))
