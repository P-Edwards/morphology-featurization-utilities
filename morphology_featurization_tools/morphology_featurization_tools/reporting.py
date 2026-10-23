"""
Writing results: tables, JSON, arrays and figures in one directory, plus a run
log saved as summary.txt. Figures are saved, and also shown when the writer is
interactive.

    report = ResultsWriter(run_dir, interactive=False)
    report.log("...")
    report.save_table(frame, "name")
    report.finish_figure(fig, "name")
    report.close()

On a machine without a display, call use_headless_backend() before importing
matplotlib.pyplot.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib


def use_headless_backend() -> None:
    """Switch matplotlib to the Agg backend; call before importing pyplot."""
    matplotlib.use("Agg")


def get_writer(
    script_path: str | None = None,
    output_dir: str | os.PathLike = "results",
    interactive: bool = False,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> "ResultsWriter":
    """
    A ResultsWriter for <output_dir>/<stem of script_path>.

    Parameters
    ----------
    script_path : str or None
        Usually __file__.
    output_dir : path
        Base results directory.
    interactive : bool
        Whether finished figures are also shown.
    formats : tuple of str
        Figure formats to write.
    """
    base = Path(output_dir)
    stem = Path(script_path).stem if script_path else "run"
    return ResultsWriter(base / stem, interactive=interactive, formats=formats)


class ResultsWriter:
    """
    Save tables, JSON, arrays, figures and a run log in a directory.

    Parameters
    ----------
    directory : path
        Destination directory; created if missing.
    interactive : bool
        If True, figures are also shown (plt.show) in addition to being saved.
    formats : tuple of str
        Figure file formats to write.
    dpi : int
        Raster figure resolution.
    """

    def __init__(
        self,
        directory: str | os.PathLike,
        interactive: bool = False,
        formats: tuple[str, ...] = ("png", "pdf"),
        dpi: int = 200,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self.formats = formats
        self.dpi = dpi
        self._log_lines: list[str] = []
        self.log(f"# Run started {datetime.now().isoformat(timespec='seconds')}")
        self.log(f"# Output directory: {self.dir.resolve()}")
        self.log(f"# Interactive: {self.interactive}")

    # -- paths ---------------------------------------------------------------

    def path(self, name: str) -> Path:
        """Path of a file in the results directory."""
        return self.dir / name

    # -- logging -------------------------------------------------------------

    def log(self, message: str = "") -> None:
        """Print a line and retain it for summary.txt."""
        print(message)
        self._log_lines.append(str(message))

    # -- tabular / structured ------------------------------------------------

    def save_table(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        """Write a DataFrame to CSV (name may omit the .csv extension)."""
        out = self.dir / (name if name.endswith(".csv") else f"{name}.csv")
        df.to_csv(out, index=index)
        self.log(f"[saved table] {out.name}  ({df.shape[0]}x{df.shape[1]})")
        return out

    def save_json(self, obj, name: str) -> Path:
        out = self.dir / (name if name.endswith(".json") else f"{name}.json")
        with open(out, "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        self.log(f"[saved json]  {out.name}")
        return out

    def save_array(self, array: np.ndarray, name: str) -> Path:
        out = self.dir / (name if name.endswith(".npy") else f"{name}.npy")
        np.save(out, np.asarray(array))
        self.log(f"[saved array] {out.name}  shape={np.asarray(array).shape}")
        return out

    # -- figures -------------------------------------------------------------

    def finish_figure(self, fig, name: str) -> list[Path]:
        """Save a figure in every configured format, then show it if
        interactive. Returns the written paths."""
        written = []
        for ext in self.formats:
            out = self.dir / f"{name}.{ext}"
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            written.append(out)
        self.log(f"[saved figure] {name}: {', '.join(p.name for p in written)}")
        if self.interactive:
            import matplotlib.pyplot as plt

            plt.show()
        return written

    def finish_open_figures(self, prefix: str) -> list[Path]:
        """Save every open figure as <prefix>_<n>, then show them all if
        interactive."""
        import matplotlib.pyplot as plt

        written: list[Path] = []
        nums = plt.get_fignums()
        for i, num in enumerate(nums):
            fig = plt.figure(num)
            name = prefix if len(nums) == 1 else f"{prefix}_{i}"
            for ext in self.formats:
                out = self.dir / f"{name}.{ext}"
                fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
                written.append(out)
        self.log(f"[saved figures] {prefix}: {len(nums)} figure(s)")
        if self.interactive:
            plt.show()
        return written

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> Path:
        """Flush the accumulated log to summary.txt."""
        out = self.dir / "summary.txt"
        with open(out, "w") as fh:
            fh.write("\n".join(self._log_lines) + "\n")
        return out


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
