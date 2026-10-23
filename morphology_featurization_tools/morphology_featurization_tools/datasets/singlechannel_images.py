import os
import numpy as np
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
from PIL import Image
import re



class FlatTifNamingConvention:
    """
    Reads class and batch labels from file names of the form
    <experiment>_<date>_<cell number>_<cell type>[_EGF].tif, e.g.
    exp1_0702_25_0_MEF_EGF.tif. The cell type is MEF or KO2; the date may be
    one token or split over several ("0702_25").
    """

    def extract_class(self, file_name: str) -> str:
        """Cell type and EGF treatment, e.g. "MEF_EGF" or "KO2_noEGF"."""
        stem = file_name.replace('.tif', '').strip()
        parts = stem.split('_')
        cell_type = next(p for p in parts if p in ('MEF', 'KO2'))
        egf = 'EGF' if 'EGF' in parts else 'noEGF'
        return f"{cell_type}_{egf}"

    def extract_batch(self, file_name: str) -> str:
        """The acquisition date with its separators removed, e.g. "070225"."""
        stem = file_name.replace('.tif', '').strip()
        parts = stem.split('_')
        # Date may be '070225' or '07', '23', '25' - rejoin date parts
        # (everything between the experiment number and the cell type token)
        cell_type_idx = next(i for i, p in enumerate(parts) if p in ('MEF', 'KO2'))
        date_parts = parts[1:cell_type_idx - 1] if cell_type_idx > 2 else parts[1:cell_type_idx]
        return ''.join(date_parts)        


class FlatTifDataset(Dataset):
    """
    Dataset for single-channel .tif images stored as flat files in a directory.
    Each __getitem__ returns {"image": (H, W) array, "class": str, "batch": str}.
    """

    def __init__(self, root_dir, convention, image_ext=".tif"):
        self.root_dir = Path(root_dir)
        self.convention = convention
        self.image_files = sorted(self.root_dir.glob(f"*{image_ext}"))
        if not self.image_files:
            raise ValueError(f"No {image_ext} files found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        image = np.array(Image.open(path)).astype(float)
        name = path.name
        return {
            "image": image,
            "class": self.convention.extract_class(name),
            "batch": self.convention.extract_batch(name),
        }