import os
import numpy as np
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
from PIL import Image
import re

class MicroscopyNamingConvention:
    """Encapsulates the three extractor functions for a specific naming convention."""

    def extract_class(self, dir_name: str) -> str:
        raise NotImplementedError

    def extract_batch(self, dir_name: str) -> str:
        raise NotImplementedError

    def extract_channel(self, file_name: str) -> str:
        raise NotImplementedError


class MultiChannelImageDataset(Dataset):
    """
    A PyTorch Dataset for multi-channel microscopy images organized in
    per-acquisition subdirectories.

    Each subdirectory contains one .tif file per channel. The dataset returns
    items as dicts with keys "image", "class", and "batch".

    Args:
        root_dir: Path to the top-level directory containing image subdirectories.
        class_extractor: Function that takes a subdirectory name (str) and returns
                         the class label string.
        batch_extractor: Function that takes a subdirectory name (str) and returns
                         the batch/date string.
        channel_extractor: Function that takes a filename (str) and returns a
                           channel name string. Channel ordering is determined by
                           the sorted channel names found in the first subdirectory,
                           and is held fixed for all subsequent directories.
        image_ext: File extension to look for when finding channel images (default ".tif").
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        class_extractor: Callable[[str], str],
        batch_extractor: Callable[[str], str],
        channel_extractor: Callable[[str], str],
        image_ext: str = ".tif",
    ):
        self.root_dir = Path(root_dir)
        self.class_extractor = class_extractor
        self.batch_extractor = batch_extractor
        self.image_ext = image_ext

        # Collect all subdirectories (each is one "sample")
        self.sample_dirs = [
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ]

        if not self.sample_dirs:
            raise ValueError(f"No subdirectories found in {self.root_dir}")

        # Build a stable channel_name -> index mapping from the first directory
        first_dir = self.sample_dirs[0]
        first_files = self._get_image_files(first_dir)
        channel_names = sorted(channel_extractor(f.name) for f in first_files)

        if len(channel_names) == 0:
            raise ValueError(f"No image files found in first directory: {first_dir}")

        self._channel_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(channel_names)
        }
        self.num_channels = len(channel_names)

        # Store a wrapped channel extractor that maps name -> stable integer index
        raw_extractor = channel_extractor
        channel_map = self._channel_to_idx

        def _channel_index(filename: str) -> int:
            name = raw_extractor(filename)
            if name not in channel_map:
                raise KeyError(
                    f"Channel '{name}' (from file '{filename}') was not seen in the "
                    f"first directory. Known channels: {list(channel_map.keys())}"
                )
            return channel_map[name]

        self._channel_index = _channel_index

    @classmethod
    def from_convention(
        cls,
        root_dir: str | os.PathLike,
        convention: MicroscopyNamingConvention,
        image_ext: str = ".tif",
    ):
        return cls(
            root_dir,
            class_extractor=convention.extract_class,
            batch_extractor=convention.extract_batch,
            channel_extractor=convention.extract_channel,
            image_ext=image_ext,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_image_files(self, directory: Path) -> list[Path]:
        """Return all image files in *directory* matching self.image_ext."""
        return list(directory.glob(f"*{self.image_ext}"))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.sample_dirs[idx]
        dir_name = sample_dir.name

        image_files = self._get_image_files(sample_dir)
        if len(image_files) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channel files in '{sample_dir}', "
                f"found {len(image_files)}."
            )

        # Read each channel file and stack the planes in channel order.
        channel_arrays: dict[int, np.ndarray] = {}
        for f in image_files:
            ch_idx = self._channel_index(f.name)
            channel_arrays[ch_idx] = np.array(Image.open(f))

        # Stack into (C, H, W)
        image = np.stack([channel_arrays[i] for i in range(self.num_channels)], axis=0)

        return {
            "image": image,                          # np.ndarray, shape (C, H, W)
            "class": self.class_extractor(dir_name),
            "batch": self.batch_extractor(dir_name),
        }


class July26NamingConvention(MicroscopyNamingConvention):
    """
    Naming convention where:
      - Batch is the leading date token (e.g. '060425')
      - Class is cell type (MEF/KO2) x EGF presence
      - Channel is extracted from '_cN_ORG' suffix
    """

    def extract_batch(self, dir_name: str) -> str:
        return dir_name.split("_")[0]

    def extract_class(self, dir_name: str) -> str:
        parts = dir_name.split("_")
        cell_type = parts[1]
        egf = "EGF" if "EGF" in parts else "noEGF"
        return f"{cell_type}_{egf}"

    def extract_channel(self, file_name: str) -> str:
        match = re.search(r"_(c\d+)_ORG", file_name)
        if not match:
            raise ValueError(f"Could not find channel token in filename: {file_name}")
        return match.group(1)        

