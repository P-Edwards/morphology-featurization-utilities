"""
Per-field feature extractors: one vector per whole field, no segmentation.

  autoencoder  OpenPhenom embeddings, on fields without illumination correction
  tda          persistence landscapes from the tdaexplore package
"""

from __future__ import annotations

import numpy as np

from .base import Regime, register_feature


@register_feature("autoencoder")
class AutoencoderFeature:
    """
    OpenPhenom MAE embeddings of whole fields, computed from a Dataset with
    extract_fields_from_dataset (the featurizer builds its own dataloader).

    Parameters
    ----------
    channel_map : list of (int or None) or None
        Field channels mapped onto model input channels (see
        autoencoder._default_channel_map). OpenPhenom is channel-agnostic, so
        one entry per real channel works; repeating a channel gives that stain
        more weight in the mean embedding.
    patch_size, batch_size, num_workers, num_features :
        Passed through to the OpenPhenom featurizer.
    """
    regime = Regime.PER_FIELD

    def __init__(
        self,
        channel_map: list[int | None] | None = None,
        patch_size: int = 256,
        batch_size: int = 128,
        num_workers: int = 0,
        num_features: int = 384,
    ):
        self.channel_map = channel_map
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_features = num_features

    def feature_names(self) -> list[str]:
        return [f"autoencoder__{i}" for i in range(self.num_features)]

    def extract_fields_from_dataset(self, dataset):
        """
        Featurize a Dataset directly with the OpenPhenom MAE.

        Returns (features, metadata) as produced by featurize_dataset: a
        (N_fields, num_features) array and a per-field metadata DataFrame.
        """
        from ..autoencoder import featurize_dataset

        return featurize_dataset(
            dataset,
            channel_map=self.channel_map,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_features=self.num_features,
        )


@register_feature("tda")
class TDAFeature:
    """
    Persistence-landscape features of whole fields from the tdaexplore
    package: patches of radius `pixel_radius_for_patches` are sampled, an
    alpha-complex persistence landscape is computed for each, and the patch
    landscapes are averaged per field.

    A landscape vector has 2 (homological degrees) x number_of_layers x 401
    entries; the grid has 401 points for any radius (tdaexplore uses
    dx = radius / 200). 10 layers give 8,020 features, tdaexplore's default
    of 50 gives 40,100; a dimensionality reduction such as PCA is usually
    applied to the block before downstream use.

    Parameters
    ----------
    channel : int
        Index into the field's channel axis; one instance per channel.
    channel_name : str | None
        Channel name for the feature prefix. None gives an unsuffixed prefix.
    pixel_radius_for_patches : int
        Patch radius in pixels.
    number_of_layers : int
        Landscape layers retained per homological degree.
    number_of_patches : int | None
        Patches per field. None lets tdaexplore choose from the field area via
        `patch_ratio`, which suits fields of different sizes.
    """
    regime = Regime.PER_FIELD

    def __init__(
        self,
        channel: int = 0,
        channel_name: str | None = None,
        pixel_radius_for_patches: int = 75,
        number_of_layers: int = 10,
        number_of_patches: int | None = None,
        patch_ratio: float = 2.0,
        proportion_of_patch_sparse: float = 0.025,
        noise_threshold: float = 0.05,
        n_jobs: int = 1,
        random_state=None,
    ):
        self.channel = channel
        self.channel_name = channel_name
        self.pixel_radius_for_patches = pixel_radius_for_patches
        self.number_of_layers = number_of_layers
        self.number_of_patches = number_of_patches
        self.patch_ratio = patch_ratio
        self.proportion_of_patch_sparse = proportion_of_patch_sparse
        self.noise_threshold = noise_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._transformer = None
        self._n_features: int | None = None

    @property
    def short(self) -> str:
        return "tda" if self.channel_name is None else f"tda_{self.channel_name}"

    def _get_transformer(self):
        if self._transformer is None:
            from tdaexplore import PatchLandscapeTransformer
            self._transformer = PatchLandscapeTransformer(
                number_of_patches=self.number_of_patches,
                pixel_radius_for_patches=self.pixel_radius_for_patches,
                proportion_of_patch_sparse=self.proportion_of_patch_sparse,
                noise_threshold=self.noise_threshold,
                number_of_layers=self.number_of_layers,
                patch_ratio=self.patch_ratio,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        return self._transformer

    def feature_names(self) -> list[str]:
        if self._n_features is None:
            return []
        return [f"{self.short}__{i}" for i in range(self._n_features)]

    def extract_fields(self, fields, metadata, n_jobs: int = 1):
        """
        Featurize fields. Returns (features, metadata) at the field level, one
        averaged landscape vector per field.
        """
        transformer = self._get_transformer()
        transformer.n_jobs = max(1, n_jobs)

        selected = [
            (f[self.channel] if f.ndim == 3 else f).astype(float) for f in fields
        ]
        if not selected:
            self._n_features = 0
            return np.zeros((0, 0)), metadata

        # One row per patch, grouped by image in input order; averaged per image.
        patch_matrix = transformer.transform(selected)
        patch_matrix = np.asarray(
            patch_matrix.todense() if hasattr(patch_matrix, "todense") else patch_matrix
        )
        per_image = getattr(transformer, "patches_per_image_", None)
        if not per_image or per_image * len(selected) != patch_matrix.shape[0]:
            # Even split when the transformer does not report patches per image.
            per_image = patch_matrix.shape[0] // max(1, len(selected))
        features = np.vstack([
            patch_matrix[i * per_image:(i + 1) * per_image].mean(axis=0)
            for i in range(len(selected))
        ])
        self._n_features = features.shape[1]
        return features, metadata
