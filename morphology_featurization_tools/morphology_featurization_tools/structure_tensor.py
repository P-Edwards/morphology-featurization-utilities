from scipy import ndimage
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Callable
from scipy.stats import skew, kurtosis


class StructureTensorTransformer(TransformerMixin, BaseEstimator):
	"""
	Sklearn-compatible transformer that computes the structure tensor for a
	batch of grayscale or multi-channel images.

	Input X:
	  - 3D (N, H, W):    one grayscale image per sample
	  - 4D (N, C, H, W): one multi-channel image per sample

	Output Y:
	  - 3D input  → (N, H, W, 3):    Y[i, :, :, 0] = Sxx
									  Y[i, :, :, 1] = Sxy
									  Y[i, :, :, 2] = Syy
	  - 4D input  → (N, C, H, W, 3): Y[i, c, :, :, 0] = Sxx for channel c
									  Y[i, c, :, :, 1] = Sxy
									  Y[i, c, :, :, 2] = Syy

	Args:
		sigma_gradient:    Sigma for Gaussian derivative (gradient smoothing).
		sigma_integration: Sigma for Gaussian smoothing of the outer products.
	"""

	def __init__(self, sigma_gradient: float = 1.0, sigma_integration: float = 2.0):
		self.sigma_gradient = sigma_gradient
		self.sigma_integration = sigma_integration

	def fit(self, X, y=None):
		return self

	def _signal_bounding_box(self, image: np.ndarray, margin: int) -> tuple[int, int, int, int]:
		"""
		Bounding box of the nonzero (foreground) pixels of `image`, expanded
		by `margin` pixels on every side and clipped to the image bounds.
		Returns (top, bottom, left, right) as half-open row/col bounds. If the
		image is entirely zero, returns the full-image box.
		"""
		foreground_rows, foreground_cols = np.nonzero(image)
		if foreground_rows.size == 0:
			return 0, image.shape[0], 0, image.shape[1]

		image_height, image_width = image.shape
		top    = max(int(foreground_rows.min()) - margin, 0)
		bottom = min(int(foreground_rows.max()) + margin + 1, image_height)
		left   = max(int(foreground_cols.min()) - margin, 0)
		right  = min(int(foreground_cols.max()) + margin + 1, image_width)
		return top, bottom, left, right

	def _structure_tensor_2d(self, image: np.ndarray) -> np.ndarray:
		"""
		Structure tensor of a single 2D image, shape (H, W, 3):
		[..., 0]=Sxx, [..., 1]=Sxy, [..., 2]=Syy.

		Crops are zero outside the cell, so the convolutions are computed only
		within the foreground's bounding box plus a margin of 3 sigma; outside
		it the result is zero.
		"""
		img = image.astype(np.float32)
		if img.max() > 0:
			img = img / img.max()

		# 3 sigma covers each Gaussian's support.
		gradient_margin    = int(np.ceil(3 * self.sigma_gradient))
		integration_margin = int(np.ceil(3 * self.sigma_integration))

		# --- gradients, restricted to the foreground box (gradient margin) ---
		grad_top, grad_bottom, grad_left, grad_right = self._signal_bounding_box(
			img, gradient_margin
		)
		gradient_window = img[grad_top:grad_bottom, grad_left:grad_right]

		Ix = np.zeros_like(img)
		Iy = np.zeros_like(img)
		Ix[grad_top:grad_bottom, grad_left:grad_right] = ndimage.gaussian_filter(
			gradient_window, sigma=self.sigma_gradient, order=[0, 1]
		)
		Iy[grad_top:grad_bottom, grad_left:grad_right] = ndimage.gaussian_filter(
			gradient_window, sigma=self.sigma_gradient, order=[1, 0]
		)

		# --- integration smoothing of the outer products (integration margin) ---
		int_top, int_bottom, int_left, int_right = self._signal_bounding_box(
			img, integration_margin
		)

		def _integrate(product: np.ndarray) -> np.ndarray:
			out = np.zeros_like(product)
			out[int_top:int_bottom, int_left:int_right] = ndimage.gaussian_filter(
				product[int_top:int_bottom, int_left:int_right],
				sigma=self.sigma_integration,
			)
			return out

		Sxx = _integrate(Ix * Ix)
		Sxy = _integrate(Ix * Iy)
		Syy = _integrate(Iy * Iy)

		return np.stack([Sxx, Sxy, Syy], axis=-1).astype(np.float32, copy=False)  # (H, W, 3)

	def transform(self, X: np.ndarray) -> np.ndarray:
		if X.ndim == 3:
			# (N, H, W) -> (N, H, W, 3)
			N, H, W = X.shape
			out = np.empty((N, H, W, 3), dtype=np.float32)
			for i in range(N):
				out[i] = self._structure_tensor_2d(X[i])
			return out

		elif X.ndim == 4:
			# (N, C, H, W) -> (N, C, H, W, 3)
			N, C, H, W = X.shape
			out = np.empty((N, C, H, W, 3), dtype=np.float32)
			for i in range(N):
				for c in range(C):
					out[i, c] = self._structure_tensor_2d(X[i, c])
			return out

		else:
			raise ValueError(f"Expected 3D or 4D input, got shape {X.shape}.")

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.transform(X)


class StructureTensorFeatureExtractor(TransformerMixin, BaseEstimator):
	"""
	Computes per-pixel structure tensor derived features from the output of
	StructureTensorTransformer.

	Input Y:
	  - 4D (N, H, W, 3):    output of StructureTensorTransformer on 3D input
	  - 5D (N, C, H, W, 3): output of StructureTensorTransformer on 4D input

	Output:
	  - 4D (N, H, W, 5):    last axis = [lambda1, lambda2, coherence,
	  - 5D (N, C, H, W, 5):               energy, eigenvalue_ratio]
	"""

	def fit(self, X, y=None):
		return self

	def _compute_features_2d(self, tensor: np.ndarray) -> np.ndarray:
		"""
		Compute derived features from a single (H, W, 3) structure tensor.
		Returns (H, W, 5).
		"""
		Sxx = tensor[..., 0]
		Sxy = tensor[..., 1]
		Syy = tensor[..., 2]

		trace = Sxx + Syy
		det = Sxx * Syy - Sxy * Sxy
		discriminant = np.sqrt(np.maximum(trace**2 - 4 * det, 0))

		lambda1 = (trace + discriminant) / 2.0
		lambda2 = (trace - discriminant) / 2.0

		coherence    = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
		energy       = lambda1 + lambda2
		eigenvalue_ratio = lambda1 / (lambda2 + 1e-10)

		return np.stack([lambda1, lambda2, coherence, energy, eigenvalue_ratio], axis=-1).astype(np.float32, copy=False)

	def transform(self, X: np.ndarray) -> np.ndarray:
		if X.ndim == 4:
			# (N, H, W, 3) -> (N, H, W, 5)
			N, H, W, _ = X.shape
			out = np.empty((N, H, W, 5), dtype=np.float32)
			for i in range(N):
				out[i] = self._compute_features_2d(X[i])
			return out

		elif X.ndim == 5:
			# (N, C, H, W, 3) -> (N, C, H, W, 5)
			N, C, H, W, _ = X.shape
			out = np.empty((N, C, H, W, 5), dtype=np.float32)
			for i in range(N):
				for c in range(C):
					out[i, c] = self._compute_features_2d(X[i, c])
			return out

		else:
			raise ValueError(f"Expected 4D or 5D input, got shape {X.shape}.")

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.transform(X)







class StructureTensorSummaryStatistics(TransformerMixin, BaseEstimator):
	"""
	Reduces the per-pixel feature maps from StructureTensorFeatureExtractor
	to per-image (or per-image-per-channel) summary statistics.

	Input:
	  - 4D (N, H, W, 5):    single-channel path
	  - 5D (N, C, H, W, 5): multi-channel path

	Output:
	  - 2D (N, F):           one feature vector per image  (4D input path)
	  - 3D (N, C, F):        one feature vector per channel (5D input path)
	  where F is the total number of (feature, stat) pairs in the summary plan.

	Stats are specified by name and resolved against STAT_REGISTRY. Every
	feature receives the stats listed in `shared_stats`. Individual features
	may receive additional stats via `feature_stats`, a mapping of feature
	name -> list of extra stat names. The output order is, for each feature
	in FEATURE_NAMES: the shared stats in order, then that feature's extra
	stats in order. Column names and computed values are both derived from a
	single ordered plan built at construction, so they stay in lockstep.

	Args:
		shared_stats: list of stat names applied to every feature.
					  Defaults to ['mean', 'std', 'p25', 'median', 'p75'].
		feature_stats: mapping of feature name -> list of additional stat
					   names for that feature only. Defaults to
					   {'coherence': ['high_fraction']}.
		retain_threshold: pixels whose ALL feature values are <= this threshold
						  are treated as background and excluded from statistics.
						  Set to None to disable masking.
		as_pandas_dataframe: if True and input is 4D (single-channel), return
							 a DataFrame with one row per image.
	"""
	FEATURE_NAMES = ['lambda1', 'lambda2', 'coherence', 'energy', 'eigenvalue_ratio']

	STAT_REGISTRY = {
		'mean':          np.mean,
		'std':           np.std,
		'median':        np.median,
		'p25':           lambda v: np.percentile(v, 25),
		'p75':           lambda v: np.percentile(v, 75),
		'high_fraction': lambda v: np.mean(v > 0.5),
	}

	# Quantile statistics, computed together in one np.percentile call.
	PERCENTILE_QUANTILES = {'p25': 25, 'median': 50, 'p75': 75}

	DEFAULT_SHARED_STATS = ['mean', 'std', 'p25', 'median', 'p75']
	DEFAULT_FEATURE_STATS = {'coherence': ['high_fraction']}

	def __init__(
		self,
		shared_stats: list[str] | None = None,
		feature_stats: dict[str, list[str]] | None = None,
		retain_threshold: float | None = 1e-5,
		as_pandas_dataframe: bool = False,
	):
		self.shared_stats = shared_stats if shared_stats is not None \
							 else self.DEFAULT_SHARED_STATS
		self.feature_stats = feature_stats if feature_stats is not None \
							 else self.DEFAULT_FEATURE_STATS
		self.retain_threshold = retain_threshold
		self.as_pandas_dataframe = as_pandas_dataframe

		# Ordered (feature_index, feature_name, stat_name, stat_callable)
		# entries; values and column names both follow this order.
		self._plan = []
		for feature_index, feature_name in enumerate(self.FEATURE_NAMES):
			stat_names = list(self.shared_stats)
			stat_names += self.feature_stats.get(feature_name, [])
			for stat_name in stat_names:
				stat_fn = self.STAT_REGISTRY[stat_name]
				self._plan.append((feature_index, feature_name, stat_name, stat_fn))

		# Output slots grouped by feature.
		self._slots_by_feature: dict[int, list[tuple[int, str, object]]] = {}
		for slot, (feature_index, _, stat_name, stat_fn) in enumerate(self._plan):
			self._slots_by_feature.setdefault(feature_index, []).append(
				(slot, stat_name, stat_fn)
			)

	@property
	def column_names_(self) -> list[str]:
		return [f"{feature_name}_{stat_name}"
				for _, feature_name, stat_name, _ in self._plan]

	def fit(self, X, y=None):
		return self

	def _summarize(self, feature_map: np.ndarray) -> np.ndarray:
		"""
		Summarise a single (H, W, F) feature map to a 1D vector whose length
		equals len(self._plan).

		Background masking: retain only pixels where at least one feature
		exceeds retain_threshold (i.e. the pixel is not uniformly zero/flat).
		"""
		H, W, F = feature_map.shape
		pixels = feature_map.reshape(-1, F)           # (H*W, F)

		if self.retain_threshold is not None:
			foreground = np.any(pixels > self.retain_threshold, axis=1)
			pixels = pixels[foreground]               # (N_fg, F)

		if len(pixels) == 0:
			return np.zeros(len(self._plan))

		out = np.empty(len(self._plan), dtype=np.float64)
		for feature_index, slots in self._slots_by_feature.items():
			values = pixels[:, feature_index]

			# One np.percentile call for all of this feature's quantile stats.
			quantile_names = [name for _, name, _ in slots
								  if name in self.PERCENTILE_QUANTILES]
			quantile_values = {}
			if quantile_names:
				computed = np.percentile(
					values,
					[self.PERCENTILE_QUANTILES[name] for name in quantile_names],
				)
				quantile_values = dict(zip(quantile_names, np.atleast_1d(computed)))

			for slot, stat_name, stat_fn in slots:
				if stat_name in quantile_values:
					out[slot] = quantile_values[stat_name]
				else:
					out[slot] = stat_fn(values)
		return out

	def transform(self, X: np.ndarray):
		if X.ndim == 4:
			# (N, H, W, F) -> (N, P); a batch may have no cells.
			if len(X) == 0:
				result = np.zeros((0, len(self._plan)), dtype=np.float32)
			else:
				result = np.stack([self._summarize(X[i]) for i in range(len(X))])
			if self.as_pandas_dataframe:
				return pd.DataFrame(result, columns=self.column_names_)
			return result

		elif X.ndim == 5:
			# (N, C, H, W, F) -> (N, C, P)
			if len(X) == 0:
				return np.zeros((0, X.shape[1], len(self._plan)), dtype=np.float32)
			return np.stack([
				np.stack([self._summarize(X[i, c]) for c in range(X.shape[1])])
				for i in range(len(X))
			])

		else:
			raise ValueError(f"Expected 4D or 5D input, got shape {X.shape}.")

	def predict(self, X: np.ndarray):
		return self.transform(X)       


class IntensityStatistics(TransformerMixin, BaseEstimator):
	"""
	Computes per-image intensity statistics on foreground pixels.

	Input X:
	  - 3D (N, H, W):    one grayscale image per sample
	  - 4D (N, C, H, W): one multi-channel image per sample

	Output:
	  - 2D (N, F)   or DataFrame: single-channel / grayscale path
	  - 3D (N, C, F):             multi-channel path

	Features (in order): mean, std, median, skew, kurtosis, foreground_fraction

	Args:
		retain_threshold: pixels at or below this value are treated as background.
		as_pandas_dataframe: if True and input is 3D, return a DataFrame.
	"""

	FEATURE_NAMES = ['intensity_mean', 'intensity_std', 'intensity_median',
					 'intensity_skew', 'intensity_kurtosis', 'foreground_fraction']

	def __init__(self, retain_threshold: float = 0.0, as_pandas_dataframe: bool = False):
		self.retain_threshold = retain_threshold
		self.as_pandas_dataframe = as_pandas_dataframe

	def fit(self, X, y=None):
		return self

	def _summarize(self, image: np.ndarray) -> np.ndarray:
		"""Compute intensity statistics for a single 2D image."""
		foreground = image[image > self.retain_threshold]

		if len(foreground) == 0:
			return np.zeros(len(self.FEATURE_NAMES))

		return np.array([
			np.mean(foreground),
			np.std(foreground),
			np.median(foreground),
			skew(foreground),
			kurtosis(foreground),
			len(foreground) / image.size,
		])

	def transform(self, X: np.ndarray):
		if X.ndim == 3:
			result = np.stack([self._summarize(X[i]) for i in range(len(X))])
			if self.as_pandas_dataframe:
				return pd.DataFrame(result, columns=self.FEATURE_NAMES)
			return result

		elif X.ndim == 4:
			# (N, C, H, W) -> (N, C, F)
			return np.stack([
				np.stack([self._summarize(X[i, c]) for c in range(X.shape[1])])
				for i in range(len(X))
			])

		else:
			raise ValueError(f"Expected 3D or 4D input, got shape {X.shape}.")

	def predict(self, X: np.ndarray):
		return self.transform(X)