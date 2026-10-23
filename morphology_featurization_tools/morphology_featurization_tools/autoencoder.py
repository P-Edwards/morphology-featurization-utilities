"""
Whole-field embeddings from the pretrained OpenPhenom masked autoencoder
(recursionpharma/OpenPhenom on Hugging Face).

Each field is cut into non-overlapping square patches; every patch is encoded
and the patch embeddings of a field are averaged into one 384-dimensional
vector.

    features, metadata = featurize_dataset(dataset)

Requires the OpenPhenom model code, which install_openphenom.sh (at the
repository root) downloads after you accept its license. The model weights
are downloaded from Hugging Face on first use and are loaded once per process.
"""

import warnings

import torch
import numpy as np
from torch.utils.data import DataLoader
try:
    from openphenom.huggingface_mae import MAEModel
except ModuleNotFoundError as error:
    if not (error.name or "").startswith("openphenom"):
        raise
    raise ModuleNotFoundError(
        "The OpenPhenom model code is not installed. It is distributed under "
        "Recursion's Non-Commercial End User License Agreement; to accept the "
        "license and download it, run install_openphenom.sh from the "
        "repository root.", name=error.name) from error
from .datasets import MultiChannelImageDataset, FlatTifDataset
import pandas as pd


# Channels OpenPhenom was trained with; the default channel map pads to this.
MODEL_CHANNELS = 6

# Most channels the encoder accepts (max_in_chans in the released config).
MAX_MODEL_CHANNELS = 11

# Channel the default map repeats when there are fewer than MODEL_CHANNELS
# inputs: index 1 is RNA in [DNA, RNA, ER, AGP, Mito]. JUMP's RNA stain also
# covers the nucleoli, which a six-channel protocol images separately.
DEFAULT_DUPLICATE_CHANNEL = 1


def _default_channel_map(n_input_channels: int) -> list[int | None]:
    """
    Default channel map for C input channels: identity, then
    DEFAULT_DUPLICATE_CHANNEL repeated up to MODEL_CHANNELS.

    Examples of explicit maps: [0, 1, 2, 3, 4] passes five channels unpadded
    (the encoder is channel-agnostic); [0, 1, 2, 3, 4, 1] repeats RNA;
    [0, 1, 2, 3, 4, None] zero-pads. Zero padding is best avoided: a zero
    plane stays zero through InstanceNorm and adds 256 empty tokens to the
    self-attention and to the mean that forms the embedding.
    """
    if n_input_channels < 1:
        raise ValueError(f"n_input_channels must be >= 1, got {n_input_channels}.")
    if n_input_channels >= MODEL_CHANNELS:
        return list(range(min(n_input_channels, MAX_MODEL_CHANNELS)))
    base = list(range(n_input_channels))
    repeat = min(DEFAULT_DUPLICATE_CHANNEL, n_input_channels - 1)
    return base + [repeat] * (MODEL_CHANNELS - n_input_channels)


def _apply_channel_map(image: np.ndarray, channel_map: list[int | None]) -> np.ndarray:
    """Map a (C, H, W) image to a (len(channel_map), H, W) image; None
    entries are zero planes."""
    n_in, H, W = image.shape
    if not 1 <= len(channel_map) <= MAX_MODEL_CHANNELS:
        raise ValueError(
            f"channel_map must have between 1 and {MAX_MODEL_CHANNELS} entries, "
            f"got {len(channel_map)}."
        )
    if all(index is None for index in channel_map):
        raise ValueError("channel_map maps no real input channels.")
    out_of_range = [i for i in channel_map if i is not None and not 0 <= i < n_in]
    if out_of_range:
        raise ValueError(
            f"channel_map refers to input channel(s) {sorted(set(out_of_range))}, "
            f"but the image has {n_in}."
        )
    output = np.zeros((len(channel_map), H, W), dtype=image.dtype)
    for out_idx, in_idx in enumerate(channel_map):
        if in_idx is not None:
            output[out_idx] = image[in_idx]
    return output


def _iter_patches(width: int, height: int, patch_size: int):
    """
    Top-left corners of the largest centred grid of whole patches, so the
    part of the field that does not fit is an even rim on opposite sides.

    The fraction of the field covered depends on the image size, which differs
    between JUMP sources (2048x2048: 100%, 1080x1080: 90%, 1224x904: 71%,
    1000x1000: 59% at patch size 256). The model needs 256x256 crops, so the
    rim cannot be avoided without resampling.
    """
    n_x, n_y = width // patch_size, height // patch_size
    if n_x == 0 or n_y == 0:
        raise ValueError(
            f"image is {width}x{height} but patch_size is {patch_size}; "
            f"no whole patch fits. The model requires {patch_size}x{patch_size} "
            f"crops, so a smaller field cannot be embedded without resampling."
        )
    off_x = (width - n_x * patch_size) // 2
    off_y = (height - n_y * patch_size) // 2
    for x in range(off_x, off_x + n_x * patch_size, patch_size):
        for y in range(off_y, off_y + n_y * patch_size, patch_size):
            yield x, y


def _patch_image(image: np.ndarray, patch_size: int) -> np.ndarray:
    """(C, H, W) -> (n_patches, C, patch_size, patch_size)"""
    _, H, W = image.shape
    patches = [
        image[:, y:y + patch_size, x:x + patch_size]
        for x, y in _iter_patches(W, H, patch_size)
    ]
    if not patches:
        raise ValueError(f"no patches extracted from a {H}x{W} image")
    return np.stack(patches)


def make_mae_dataloader(
    dataset: MultiChannelImageDataset | FlatTifDataset,
    channel_map: list[int | None] | None = None,
    patch_size: int = 256,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap an image dataset in a DataLoader for MAE inference.

    Each batch is a tuple of
      patches               torch.Tensor (total_patches, C', patch_size, patch_size)
      metadata              list of dicts, every non-image key of each item
      n_patches_per_image   list[int]

    Args:
        dataset:     any dataset whose items have an "image" of shape (C, H, W)
                     or (H, W).
        channel_map: 1 to MAX_MODEL_CHANNELS entries, each an input channel
                     index or None (zero plane). Default: _default_channel_map.
        patch_size:  side of each square patch (default 256).
        batch_size:  images per batch (default 128).
        num_workers: DataLoader worker processes (default 0).
        shuffle:     whether to shuffle the dataset (default False).
    """

    if channel_map is None:
        first_image = dataset[0]["image"]
        n_channels = 1 if first_image.ndim == 2 else first_image.shape[0]
        channel_map = _default_channel_map(n_channels)

    if not 1 <= len(channel_map) <= MAX_MODEL_CHANNELS:
        raise ValueError(
            f"channel_map must have between 1 and {MAX_MODEL_CHANNELS} entries, "
            f"got {len(channel_map)}."
        )
    if all(index is None for index in channel_map):
        raise ValueError("channel_map maps no real input channels.")

    def collate_fn(batch):
        all_patches = []
        metas = []
        n_patches_per_image = []

        for item in batch:
            image = item["image"]
            if image.ndim == 2:
                image = image[np.newaxis, :, :]
            image = _apply_channel_map(image, channel_map)
            patches = _patch_image(image, patch_size)

            all_patches.append(patches)
            metas.append({k: v for k, v in item.items() if k != "image"})
            n_patches_per_image.append(len(patches))

        all_patches = torch.from_numpy(np.concatenate(all_patches, axis=0)).float()
        return all_patches, metas, n_patches_per_image

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )



def mae_inference(
    dataloader: DataLoader,
    model: MAEModel,
    num_features: int = 384,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Embed every image of a dataloader built by make_mae_dataloader.

    Runs on the GPU when one is available. A field's embedding is the mean of
    its patch embeddings.

    Args:
        dataloader:   from make_mae_dataloader.
        model:        a loaded OpenPhenom MAEModel.
        num_features: embedding width (384 for OpenPhenom).

    Returns:
        X:        (n_images, num_features) float32 array, in dataset order.
                  Rows of images that produced no embedding are NaN.
        metadata: DataFrame with one row per image, holding every non-image
                  key of the dataset items.
    """

    cuda_available = torch.cuda.is_available()
    n_images = len(dataloader.dataset)

    # NaN rather than zeros, so an image without an embedding cannot pass for
    # data (zero rows would all be identical under cosine similarity).
    X = np.full((n_images, num_features), np.nan, dtype=np.float32)
    all_meta = []
    img_idx = 0

    model.eval()
    if cuda_available:
        model.cuda()
    with torch.no_grad():
        for patches, metas, n_patches_per_image in dataloader:
            if cuda_available:
                latent = model.predict(patches.cuda()).cpu().numpy()
            else:
                latent = model.predict(patches).numpy()

            patch_idx = 0
            for n_patches in n_patches_per_image:
                X[img_idx] = latent[patch_idx:patch_idx + n_patches].mean(axis=0)
                patch_idx += n_patches
                img_idx += 1

            all_meta.extend(metas)

    unwritten = int(np.isnan(X).all(axis=1).sum())
    if unwritten:
        warnings.warn(
            f"{unwritten}/{n_images} field(s) produced no embedding and remain "
            f"NaN. They must be dropped, not zero-filled, before any cosine "
            f"retrieval.", RuntimeWarning, stacklevel=2)

    metadata = pd.DataFrame(all_meta)
    return X, metadata


def featurize_dataset(
    dataset: MultiChannelImageDataset | FlatTifDataset,
    model: MAEModel | None = None,
    channel_map: list[int | None] | None = None,
    patch_size: int = 256,
    batch_size: int = 128,
    num_workers: int = 0,
    num_features: int = 384,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Embed every image of a dataset with OpenPhenom.

    Args:
        dataset:      any dataset whose items have an "image" of shape
                      (C, H, W) or (H, W); the other keys of each item are
                      returned as metadata.
        model:        an OpenPhenom MAEModel; default the pretrained model,
                      downloaded on first use.
        channel_map:  image channels to pass to the model, as indices into
                      the image's channel axis (None gives a zero plane).
                      Default: every channel, with one channel repeated up to
                      six inputs when there are fewer (_default_channel_map).
        patch_size:   side of each square patch in pixels (default 256).
        batch_size:   images per forward pass (default 128).
        num_workers:  DataLoader worker processes (default 0).
        num_features: embedding width (384 for OpenPhenom).

    Returns:
        X:        (n_images, num_features) float32 array, in dataset order.
        metadata: DataFrame with one row per image.
    """
    if model is None:
        model = _shared_model()

    dataloader = make_mae_dataloader(
        dataset,
        channel_map=channel_map,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return mae_inference(dataloader, model, num_features=num_features)



_MODEL_CACHE: dict[str, MAEModel] = {}


def _shared_model(name: str = "recursionpharma/OpenPhenom") -> MAEModel:
    """The pretrained MAE, loaded once per process."""
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = MAEModel.from_pretrained(name)
    return _MODEL_CACHE[name]


# The normalizers are also importable from this module.
from .normalizers import TypicalVariationNormalization, MADRobustize  # noqa: E402,F401
