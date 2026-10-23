"""
diagnostics.py

Diagnostic visualizations for image-based profiling, following Caicedo et al.
2017 ("Data-analysis strategies for image-based cell profiling").

Provided:

  * plate_layout_heatmap - render a per-well quantity (e.g. cell count, or any
    feature's well-mean) in plate geometry (rows x columns), to reveal
    plate-layout effects such as edge effects and row/column gradients
    (Caicedo Fig 4b).

  * control_correlation_matrix - correlation heatmap of control-well profiles
    across plates, sorted by plate, to reveal batch effects as blocks of high
    within-plate / cross-plate correlation (Caicedo Fig 4a).

Both return a matplotlib Figure and never write to disk. Well identifiers are
parsed with the standard convention: a leading letter (or letters) as the row
and trailing digits as the column, e.g. "A02" -> row A (1), column 2; "AB12"
-> row 28, column 12.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def parse_well_id(well: str) -> tuple[int, int]:
    """
    Parse a well identifier into (row_index, column_index), both 1-based.

    Row letters are interpreted base-26 (A=1, ..., Z=26, AA=27, AB=28, ...).
    Raises ValueError if the identifier does not match the letter+digits form.
    """
    match = _WELL_RE.match(str(well).strip())
    if not match:
        raise ValueError(f"Cannot parse well id '{well}'.")
    letters, digits = match.group(1).upper(), match.group(2)

    row = 0
    for ch in letters:
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row, int(digits)


def plate_layout_heatmap(
    metadata: pd.DataFrame,
    value_column: str,
    well_column: str = "Metadata_Well",
    plate_column: str = "Metadata_Plate",
    title: str | None = None,
):
    """
    Render per-well values in plate geometry, one subplot per plate.

    Parameters
    ----------
    metadata : pd.DataFrame
        One row per well (or per field, values are averaged per well),
        containing the well id, plate id, and the value column.
    value_column : str
        Column to visualize (e.g. "n_cells" or a feature column).
    well_column, plate_column : str
        Column names for the well and plate identifiers.
    title : str, optional
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plates = list(pd.unique(metadata[plate_column]))
    n_plates = len(plates)
    ncols = min(n_plates, 3)
    nrows = int(np.ceil(n_plates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, plate in enumerate(plates):
        ax = axes[idx // ncols][idx % ncols]
        sub = metadata[metadata[plate_column] == plate]

        # Average the value per well (handles per-field input).
        per_well = sub.groupby(well_column)[value_column].mean()

        rows_cols = {w: parse_well_id(w) for w in per_well.index}
        max_row = max(rc[0] for rc in rows_cols.values())
        max_col = max(rc[1] for rc in rows_cols.values())

        grid = np.full((max_row, max_col), np.nan)
        for well, value in per_well.items():
            r, c = rows_cols[well]
            grid[r - 1, c - 1] = value

        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_title(f"{plate}", fontsize=10)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_yticks(range(max_row))
        ax.set_yticklabels([chr(ord("A") + i) if i < 26 else str(i + 1)
                            for i in range(max_row)], fontsize=7)
        ax.set_xticks(range(max_col))
        ax.set_xticklabels(range(1, max_col + 1), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes.
    for idx in range(n_plates, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title or f"Plate-layout effect: {value_column}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def control_correlation_matrix(
    features: np.ndarray,
    metadata: pd.DataFrame,
    control_mask: np.ndarray,
    plate_column: str = "Metadata_Plate",
    title: str | None = None,
):
    """
    Correlation heatmap of control-well profiles across plates.

    Control wells are ordered by plate so that batch effects appear as blocks
    of elevated within-plate correlation relative to cross-plate correlation
    (Caicedo Fig 4a).

    Parameters
    ----------
    features : np.ndarray, shape (N_wells, F)
        Well-level profiles, row-aligned with metadata.
    metadata : pd.DataFrame, shape (N_wells, ...)
        Must contain the plate column, row-aligned with features.
    control_mask : np.ndarray of bool, shape (N_wells,)
        True for control wells.
    plate_column : str
        Column identifying the plate.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    control_features = features[control_mask]
    control_plates = metadata.loc[control_mask, plate_column].to_numpy()

    order = np.argsort(control_plates, kind="stable")
    ordered_features = control_features[order]
    ordered_plates = control_plates[order]

    # Correlation between control-well profiles.
    corr = np.corrcoef(ordered_features)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title or "Control-well profile correlation (sorted by plate)",
                 fontsize=12)
    ax.set_xlabel("control well (sorted by plate)")
    ax.set_ylabel("control well (sorted by plate)")

    # Mark plate boundaries.
    boundaries = np.where(ordered_plates[:-1] != ordered_plates[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.5)
        ax.axvline(b, color="black", linewidth=0.5)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    return fig
