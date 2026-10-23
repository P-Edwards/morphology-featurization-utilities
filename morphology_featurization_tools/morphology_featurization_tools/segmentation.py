"""
segmentation.py

Cellpose-based cell segmenter for batches of microscopy images.

Takes a list of per-sample dicts as produced by iterating a DataLoader
over FlatTifDataset or MultiChannelImageDataset (without a collate_fn)
and returns a tuple of (cell_images, metadata) where cell_images is a
stacked numpy array of fixed-size crops and metadata is a pandas
DataFrame with one row per cell.

All metadata fields from the parent image are propagated to every child
cell. Each cell crop is a fixed-size canvas, zero-padded where necessary,
with the cell mask centroid centered on the canvas and the background
outside the mask zeroed.
"""

import numpy as np
import pandas as pd
from cellpose import models


class CropSegmenter:
    """
    Segment a batch of microscopy images using a Cellpose pretrained model
    and return one fixed-size crop per detected cell.

    Input is a list of per-sample dicts, each with at minimum:
        "image" : np.ndarray, shape (H, W) or (C, H, W)
        plus arbitrary metadata keys with scalar values.

    Output is a tuple (cell_images, metadata) where:
        cell_images : np.ndarray, shape (N_cells, H, W) or
                      (N_cells, C, H, W)
        metadata    : pd.DataFrame, shape (N_cells, n_meta_keys),
                      with one row per cell and metadata values
                      duplicated from each cell's parent image.

    Cellpose receives a channels-first (3, H, W) array for each image,
    constructed from up to three user-specified channel indices.
    Unused channel slots are zero-filled. channel_axis=0 is passed to
    model.eval() to match this convention. The resulting 2D masks are
    applied back to the full multi-channel image for cropping.

    Parameters
    ----------
    canvas_size : int
        Side length (pixels) of the square output crop for each cell.
    cellpose_channels : list of (int or None)
        Up to three channel indices (into the image's channel axis) to
        pass to Cellpose, e.g. [0, 1, None]. Must have length <= 3.
        None entries are zero-filled. For single-channel (H, W) inputs,
        pass [0, None, None] or similar.
    pretrained_model : str
        Cellpose pretrained model name. Default "cpsam_v2".
    gpu : bool
        Whether to run Cellpose on GPU. Default False.
    diameter : float or None
        Expected cell diameter in pixels. None uses the model default
        (diam_mean = 30px). Default None.
    flow_threshold : float
        Cellpose flow error threshold. Default 0.4.
    cellprob_threshold : float
        Cellpose cell probability threshold. Default 0.0.
    normalize : bool or dict
        Cellpose normalization argument. Default True.
    min_size : int
        Minimum cell size in pixels; smaller detections are discarded
        by Cellpose. Default 15.
    crop_channels : list of int or None
        Channel indices (into the input field) kept in the per-cell crops;
        None keeps all. The crops' channel axis is renumbered: with
        crop_channels=[3] the crops have one channel, at index 0. Affects
        segment() only; segment_fields() always returns whole fields.
    cellpose_batch_size : int
        256x256 tiles Cellpose passes through the network at once (its own
        batch_size; the library default is 8, its A100 benchmarks use 32).
        Default 32.

    Attributes (set after segment)
    ----------
    n_cells_found_ : int
        Total number of cells detected across all images in the last
        segment call.
    n_cells_discarded_ : int
        Number of cells discarded because their bounding box exceeded
        canvas_size.
    n_images_empty_ : int
        Number of input images for which Cellpose found no cells.
    cells_per_image_ : list of int
        Number of retained cells per input image, in input order.
    """

    def __init__(
        self,
        canvas_size: int,
        cellpose_channels: list[int | None],
        pretrained_model: str = "cpsam_v2",
        gpu: bool = False,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        normalize: bool | dict = True,
        min_size: int = 15,
        cellpose_batch_size: int = 32,
        crop_channels: list[int] | None = None,
    ):
        if len(cellpose_channels) > 3:
            raise ValueError(
                f"cellpose_channels must have length <= 3, got {len(cellpose_channels)}"
            )

        self.canvas_size         = canvas_size
        self.cellpose_channels   = cellpose_channels
        self.pretrained_model    = pretrained_model
        self.gpu                 = gpu
        self.diameter            = diameter
        self.flow_threshold      = flow_threshold
        self.cellprob_threshold  = cellprob_threshold
        self.normalize           = normalize
        self.min_size            = min_size
        self.cellpose_batch_size = cellpose_batch_size
        self.crop_channels = crop_channels

        # Loaded on first use.
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = models.CellposeModel(
                gpu              = self.gpu,
                pretrained_model = self.pretrained_model,
            )
        return self._model

    # ------------------------------------------------------------------
    # Channel preparation
    # ------------------------------------------------------------------

    def _select_crop_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Keep only the channels the crop-based features will read.

        Applied per field before crops are cut, so the crop stack is sized by
        the channels actually used rather than by everything the dataset
        fetched. Costs one field-sized temporary and saves that same factor on
        every cell in the batch.
        """
        if self.crop_channels is None or image.ndim != 3:
            return image
        return image[self.crop_channels]

    def _to_cellpose_input(self, image: np.ndarray) -> np.ndarray:
        """
        Build a 3-slot channels-first (3, H, W) array for Cellpose,
        selecting up to 3 channels by index and zero-filling unused slots.
        Native axis ordering is preserved; channel_axis=0 is passed to
        model.eval() so no transpose is required.

        Accepts (H, W) or (C, H, W) input.
        """
        if image.ndim == 2:
            # Single-channel: treat as (1, H, W)
            image = image[np.newaxis, :, :]

        C, H, W = image.shape
        out = np.zeros((3, H, W), dtype=image.dtype)

        for slot, ch_idx in enumerate(self.cellpose_channels):
            if ch_idx is None:
                continue
            if ch_idx >= C:
                raise ValueError(
                    f"Channel index {ch_idx} out of range for image with "
                    f"{C} channel(s)."
                )
            out[slot] = image[ch_idx]

        return out

    # ------------------------------------------------------------------
    # Crop extraction
    # ------------------------------------------------------------------

    def _extract_cell_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        label_id: int,
    ) -> np.ndarray | None:
        """
        Extract a fixed-size canvas crop for one labeled cell.

        The centroid of the cell mask is placed at the center of the
        canvas. Background outside the mask is zeroed. Returns None if
        the cell's bounding box exceeds canvas_size in either dimension.

        Parameters
        ----------
        image : np.ndarray
            Full image, shape (H, W) or (C, H, W).
        mask : np.ndarray
            2D integer label array, shape (H, W).
        label_id : int
            The label value for this cell in mask.

        Returns
        -------
        np.ndarray of shape (H, W) or (C, H, W) matching image, or None.
        """
        cell_mask = mask == label_id          # (H, W) bool

        # Bounding box check — discard if larger than canvas
        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cell_h = rmax - rmin + 1
        cell_w = cmax - cmin + 1

        if cell_h > self.canvas_size or cell_w > self.canvas_size:
            return None

        # Centroid (center of mass of mask pixels)
        coords = np.argwhere(cell_mask)       # (K, 2): rows, cols
        centroid_r = int(np.round(coords[:, 0].mean()))
        centroid_c = int(np.round(coords[:, 1].mean()))

        half   = self.canvas_size // 2
        is_odd = self.canvas_size % 2

        # Source window in the original image (may extend outside bounds)
        src_r0 = centroid_r - half
        src_c0 = centroid_c - half
        src_r1 = centroid_r + half + is_odd
        src_c1 = centroid_c + half + is_odd

        H = mask.shape[0]
        W = mask.shape[1]

        # Destination offsets on the canvas (handle out-of-bounds padding)
        dst_r0 = max(0, -src_r0)
        dst_c0 = max(0, -src_c0)
        dst_r1 = self.canvas_size - max(0, src_r1 - H)
        dst_c1 = self.canvas_size - max(0, src_c1 - W)

        # Clamp source to image bounds
        src_r0_c = max(0, src_r0)
        src_c0_c = max(0, src_c0)
        src_r1_c = min(H, src_r1)
        src_c1_c = min(W, src_c1)

        multichannel = image.ndim == 3
        C = image.shape[0] if multichannel else 1

        if multichannel:
            canvas = np.zeros((C, self.canvas_size, self.canvas_size),
                              dtype=image.dtype)
            canvas[:, dst_r0:dst_r1, dst_c0:dst_c1] = (
                image[:, src_r0_c:src_r1_c, src_c0_c:src_c1_c]
            )
        else:
            canvas = np.zeros((self.canvas_size, self.canvas_size),
                              dtype=image.dtype)
            canvas[dst_r0:dst_r1, dst_c0:dst_c1] = (
                image[src_r0_c:src_r1_c, src_c0_c:src_c1_c]
            )

        # Build the mask canvas and zero background
        mask_canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=bool)
        mask_canvas[dst_r0:dst_r1, dst_c0:dst_c1] = (
            cell_mask[src_r0_c:src_r1_c, src_c0_c:src_c1_c]
        )

        if multichannel:
            canvas[:, ~mask_canvas] = 0
        else:
            canvas[~mask_canvas] = 0

        return canvas

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def segment_masks(self, batch: list[dict]) -> list[np.ndarray]:
        """
        Run Cellpose on a batch and return the raw 2D label masks, one per
        input image, without cropping. Useful for visual inspection of
        segmentation quality. Uses the same model and parameters as
        segment(), so masks match what segment() would crop from.

        Parameters
        ----------
        batch : list of dict
            Each element is a per-sample dict with at minimum:
                "image" : np.ndarray, shape (H, W) or (C, H, W)

        Returns
        -------
        list of np.ndarray
            One (H, W) integer label array per input image.
        """
        cellpose_inputs = [
            self._to_cellpose_input(batch[image_index]["image"])
            for image_index in range(len(batch))
        ]

        # Cellpose batches tiles across images only when given one 4D array
        # (N, 3, H, W, channel_axis=1); a list is processed image by image.
        # Fields are stacked when they all have the same shape.
        shapes = {array.shape for array in cellpose_inputs}
        stackable = len(shapes) == 1 and len(cellpose_inputs) > 1
        model_input = np.stack(cellpose_inputs) if stackable else cellpose_inputs

        model = self._get_model()
        masks_list, _, _ = model.eval(
            model_input,
            diameter           = self.diameter,
            flow_threshold     = self.flow_threshold,
            cellprob_threshold = self.cellprob_threshold,
            normalize          = self.normalize,
            min_size           = self.min_size,
            batch_size         = self.cellpose_batch_size,
            channel_axis       = 1 if stackable else 0,
        )
        # Masks come back as one (N, H, W) array or a list of (H, W).
        if stackable:
            return [np.asarray(masks_list[i]) for i in range(len(cellpose_inputs))]
        return list(masks_list)

    def segment(
        self,
        batch: list[dict],
        masks: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Segment a list of per-sample image dicts and return per-cell
        crops alongside a metadata DataFrame.

        Parameters
        ----------
        batch : list of dict
            Each element is a per-sample dict with at minimum:
                "image" : np.ndarray, shape (H, W) or (C, H, W)
            plus arbitrary metadata keys with scalar values.
        masks : list of np.ndarray, optional
            Precomputed (H, W) label masks aligned with `batch`. Cellpose is
            the expensive GPU step, so a caller that needs both crops and
            field+mask outputs for the same batch (as the orchestrator does)
            can segment once and pass the masks to both, instead of running
            the model twice over the same images.

        Returns
        -------
        cell_images : np.ndarray, shape (N_cells, H, W) or
                      (N_cells, C, H, W)
        metadata    : pd.DataFrame, shape (N_cells, n_meta_keys)
        """
        number_of_images = len(batch)

        # Metadata keys (everything except "image")
        meta_keys = [k for k in batch[0].keys() if k != "image"]

        masks_list = self.segment_masks(batch) if masks is None else masks

        # Accumulate results
        out_images = []
        out_meta   = {k: [] for k in meta_keys}

        n_cells_found     = 0
        n_cells_discarded = 0
        n_images_empty    = 0
        cells_per_image   = []

        for image_index in range(number_of_images):
            image     = self._select_crop_channels(batch[image_index]["image"])
            mask      = masks_list[image_index]     # (H, W) integer labels
            label_ids = np.unique(mask)
            label_ids = label_ids[label_ids != 0]

            n_cells_found += len(label_ids)

            masks_retained = 0
            for label_id in label_ids:
                crop = self._extract_cell_crop(image, mask, int(label_id))
                if crop is None:
                    n_cells_discarded += 1
                    continue
                out_images.append(crop)
                for k in meta_keys:
                    out_meta[k].append(batch[image_index][k])
                masks_retained += 1

            if len(label_ids) == 0:
                n_images_empty += 1

            cells_per_image.append(masks_retained)

        # Diagnostic attributes
        self.n_cells_found_     = n_cells_found
        self.n_cells_discarded_ = n_cells_discarded
        self.n_images_empty_    = n_images_empty
        self.cells_per_image_   = cells_per_image

        if not out_images:
            sample_image = self._select_crop_channels(batch[0]["image"])
            empty_shape  = (
                (0, sample_image.shape[0], self.canvas_size, self.canvas_size)
                if sample_image.ndim == 3
                else (0, self.canvas_size, self.canvas_size)
            )
            return (
                np.zeros(empty_shape, dtype=sample_image.dtype),
                pd.DataFrame(columns=meta_keys),
            )

        return (
            np.stack(out_images, axis=0),
            pd.DataFrame(out_meta),
        )


def _default_aggregator(features: np.ndarray) -> np.ndarray:
    """Elementwise median across cells in a group."""
    return np.median(features, axis=0)


# Alias of CropSegmenter.
CellposeSegmenter = CropSegmenter


class FieldMaskSegmenter(CropSegmenter):
    """
    A CropSegmenter that also returns whole fields with their Cellpose label
    masks (segment_fields), the input of the cp_measure field+mask extractors.
    segment() still returns crops, so one instance serves both.
    """

    def segment_fields(
        self,
        batch: list[dict],
        masks: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
        """
        Segment a list of per-sample image dicts and return whole fields with
        their label masks and per-field metadata.

        Parameters
        ----------
        batch : list of dict
            Each element is a per-sample dict with at minimum:
                "image" : np.ndarray, shape (H, W) or (C, H, W)
            plus arbitrary metadata keys with scalar values.

        Returns
        -------
        fields : list of np.ndarray
            The input images, unchanged, one per field. Shapes (H, W) or
            (C, H, W).
        masks : list of np.ndarray
            The (H, W) integer Cellpose label mask for each field, aligned
            with `fields`. Labels are made sequential (1..K, background 0)
            so they satisfy cp_measure's contiguous-label requirement.
        metadata : pd.DataFrame
            One row per field, carrying all non-image metadata keys plus a
            "n_cells" column giving the number of labelled cells in that
            field's mask.

        Diagnostic attributes (n_cells_found_, n_images_empty_,
        cells_per_image_) are set as on segment().
        """
        number_of_images = len(batch)
        meta_keys = [k for k in batch[0].keys() if k != "image"]

        masks_list = self.segment_masks(batch) if masks is None else masks

        fields: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        meta_rows: list[dict] = []

        n_cells_found = 0
        n_images_empty = 0
        cells_per_image: list[int] = []

        for image_index in range(number_of_images):
            image = batch[image_index]["image"]
            mask = masks_list[image_index]

            # Make labels sequential (cp_measure requires contiguous labels).
            label_ids = np.unique(mask)
            label_ids = label_ids[label_ids != 0]
            sequential = np.zeros_like(mask, dtype=np.int32)
            for new_label, old_label in enumerate(label_ids, start=1):
                sequential[mask == old_label] = new_label

            n_here = len(label_ids)
            n_cells_found += n_here
            if n_here == 0:
                n_images_empty += 1
            cells_per_image.append(n_here)

            fields.append(image)
            masks.append(sequential)

            row = {k: batch[image_index][k] for k in meta_keys}
            row["n_cells"] = n_here
            meta_rows.append(row)

        self.n_cells_found_ = n_cells_found
        self.n_images_empty_ = n_images_empty
        self.cells_per_image_ = cells_per_image

        return fields, masks, pd.DataFrame(meta_rows)


class FeatureAggregator:
    """
    Aggregate per-cell feature vectors by metadata grouping keys.

    Takes a feature array and a metadata DataFrame (as returned by
    CropSegmenter.segment followed by featurization) and produces
    one aggregated feature vector per unique combination of the
    specified grouping keys.

    Parameters
    ----------
    grouping_keys : list of str
        Metadata column names to group by, e.g. ["class", "batch"].
    aggregator_fn : callable, optional
        Function with signature fn(subarray: np.ndarray) -> np.ndarray,
        where subarray has shape (N_cells_in_group, F) and the return
        value has shape (F,). Defaults to elementwise median.

    """

    def __init__(
        self,
        grouping_keys: list[str],
        aggregator_fn=None,
    ):
        self.grouping_keys  = grouping_keys
        self.aggregator_fn  = aggregator_fn if aggregator_fn is not None \
                              else _default_aggregator

    def aggregate(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate per-cell features by grouping keys.

        Parameters
        ----------
        features : np.ndarray, shape (N_cells, F)
            Per-cell feature vectors.
        metadata : pd.DataFrame, shape (N_cells, n_meta_keys)
            Per-cell metadata, must contain all columns in
            self.grouping_keys.

        Returns
        -------
        aggregated_features : np.ndarray, shape (N_groups, F)
            One aggregated feature vector per group.
        group_labels : pd.DataFrame, shape (N_groups, len(grouping_keys))
            Group identity for each row of aggregated_features,
            containing only the grouping key columns.
        """
        missing = [k for k in self.grouping_keys if k not in metadata.columns]
        if missing:
            raise ValueError(
                f"Grouping key(s) {missing} not found in metadata columns: "
                f"{list(metadata.columns)}"
            )

        # groupby().groups gives index labels; a fresh RangeIndex makes them
        # row positions into `features`.
        metadata = metadata.reset_index(drop=True)
        groups        = metadata.groupby(self.grouping_keys, sort=False)
        agg_rows      = []
        label_rows    = []

        if len(metadata) != len(features):
            raise ValueError(
                f"features has {len(features)} row(s) but metadata has "
                f"{len(metadata)}; they must correspond row for row."
            )

        for group_key, group_indices in groups.groups.items():
            subarray = features[group_indices.to_numpy()]
            agg_rows.append(self.aggregator_fn(subarray))

            # Normalise scalar key to tuple for uniform handling
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label_rows.append(dict(zip(self.grouping_keys, group_key)))

        aggregated_features = np.stack(agg_rows, axis=0)
        group_labels        = pd.DataFrame(label_rows)

        return aggregated_features, group_labels
