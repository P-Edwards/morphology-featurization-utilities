"""
Streaming PyTorch datasets over JUMP Cell Painting (cpg0016) images.

JUMPCellPaintingDataset is built from a site table: one row per site (field
of view) with SITE_KEYS, Metadata_JCP2022, and a URL_Orig<channel> column for
every channel it reads. Each item is one site; __getitem__ fetches the site's
channels from S3 and stacks them into a (C, H, W) array in the order of
`channels`. Nothing is cached to disk.

The other datasets compute their site table from the JUMP metadata tables
and the image index:

  JUMPTarget2Dataset          every well of the Target-2 plates
  JUMPCompoundDataset         the wells of given compounds on compound-screen
                              plates, and the negative controls of those plates
  JUMPPositiveControlDataset  JUMPCompoundDataset for the compound positive
                              controls
  JUMPCRISPRDataset           the wells of given genes' CRISPR guides, and the
                              negative controls of those plates

Controls (positive and negative, per modality) are read from JUMP's
perturbation_control table (control_ids).

Compounds and genes can be given in any form broad_babel translates to a
JCP2022 id: JCP2022 ids, InChIKeys or gene symbols (broad_babel's
standard_key), Broad sample ids, or NCBI gene ids. JCP2022 ids are used as
they are, without broad_babel.

Channels default to FLUORESCENT_CHANNELS, the five Cell Painting channels in
the order of the protocol (Bray et al. 2017, GigaScience 6:giw014, Table 1),
which are the image columns of the JUMP image index. The metadata-built
datasets drop sites that lack a requested channel, with a warning;
JUMPCellPaintingDataset rejects them.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

__all__ = [
    "JUMPCellPaintingDataset",
    "JUMPTarget2Dataset",
    "JUMPCompoundDataset",
    "JUMPPositiveControlDataset",
    "JUMPCRISPRDataset",
    "site_table_for_wells",
    "resolve_perturbations",
    "load_metadata",
    "control_ids",
    "DMSO_JCP2022",
    "FLUORESCENT_CHANNELS",
    "COMPOUND_PLATE_TYPES",
]

DMSO_JCP2022 = "JCP2022_033924"

# JUMP's table of control perturbations (Metadata_JCP2022, Metadata_pert_type
# poscon/negcon/empty, Metadata_Name, Metadata_modality). jump_portrait does not
# fetch it, so it is read from the jump-cellpainting/datasets repository at a
# fixed commit, as jump_portrait does for the other tables.
PERTURBATION_CONTROL_URL = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/"
    "016e865fa0691244e0860943e41c7d6a88ed2580/metadata/perturbation_control.csv"
)
PERTURBATION_CONTROL_SHA256 = (
    "f513c4017555efa486d07c3e9924944d5b0c44d2f2e76ca6e1fc934f5f718ef6"
)

FLUORESCENT_CHANNELS = ("DNA", "ER", "RNA", "AGP", "Mito")

# Plate types of the compound screen: every type except ORF and CRISPR.
COMPOUND_PLATE_TYPES = ("COMPOUND", "COMPOUND_EMPTY", "DMSO", "POSCON8",
                        "TARGET1", "TARGET2")

WELL_KEYS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
SITE_KEYS = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate",
             "Metadata_Well", "Metadata_Site"]
CONTROL_SCOPES = ("plates", "batches", "all", "none")


def url_column(channel: str) -> str:
    return f"URL_Orig{channel}"


def resolve_channels(channels: Iterable[str] | None = None) -> list[str]:
    """`channels` as a list, FLUORESCENT_CHANNELS if None."""
    resolved = list(FLUORESCENT_CHANNELS if channels is None else channels)
    if not resolved:
        raise ValueError("No channels requested.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Channels requested more than once: {resolved}")
    return resolved


# Metadata

def load_metadata(name: str) -> pd.DataFrame:
    """A JUMP metadata table: well, plate, compound, crispr or orf, as fetched
    and cached by jump_portrait, or perturbation_control."""
    if name == "perturbation_control":
        import pooch

        return pd.read_csv(pooch.retrieve(PERTURBATION_CONTROL_URL,
                                          known_hash=f"sha256:{PERTURBATION_CONTROL_SHA256}"))
    from jump_portrait.fetch import get_table

    return pd.read_csv(get_table(name))


def control_ids(pert_type: str, modality: str) -> list[str]:
    """JCP2022 ids of one kind of control ("poscon" or "negcon") in one
    modality ("compound", "crispr" or "orf"), from the perturbation_control
    table."""
    table = load_metadata("perturbation_control")
    chosen = table[(table["Metadata_pert_type"] == pert_type)
                   & (table["Metadata_modality"] == modality)]
    if chosen.empty:
        raise ValueError(f"The perturbation_control table has no {pert_type} "
                         f"for {modality}.")
    return sorted(chosen["Metadata_JCP2022"])


def site_table_for_wells(wells: pd.DataFrame,
                         channels: Iterable[str] | None = None) -> pd.DataFrame:
    """
    One row per site of `wells` (a frame with Metadata_Source, Metadata_Plate,
    Metadata_Well and Metadata_JCP2022), with SITE_KEYS, Metadata_JCP2022 and
    the URL column of each channel. Only the plates of `wells` are read from
    the image index; the filter is pushed down to the parquet reader, so the
    full index is never loaded. Sites are in index order.
    """
    import pyarrow.parquet as pq
    from jump_portrait.fetch import get_index_file

    channels = resolve_channels(channels)
    url_columns = [url_column(c) for c in channels]
    index_path = get_index_file()

    available = set(pq.read_schema(index_path).names)
    absent = [c for c in url_columns if c not in available]
    if absent:
        raise ValueError(
            f"The JUMP image index has no column {', '.join(absent)}, so "
            f"those channels cannot be fetched. Index URL columns: "
            f"{sorted(c for c in available if c.startswith('URL_'))}")

    wells = wells[WELL_KEYS + ["Metadata_JCP2022"]].drop_duplicates(WELL_KEYS)
    index_table = pq.read_table(
        index_path,
        columns=SITE_KEYS + url_columns,
        filters=[("Metadata_Plate", "in", set(wells["Metadata_Plate"]))],
    ).to_pandas()

    # The inner join attaches Metadata_JCP2022 and drops the sites of wells
    # not asked for.
    return index_table.merge(wells, on=WELL_KEYS, how="inner").reset_index(drop=True)


_JCP2022_ID = re.compile(r"JCP2022_\d+")
_BABEL_COLUMNS = ("standard_key", "broad_sample", "NCBI_Gene_ID", "JCP2022")


def _same_identifier(found, entry: str) -> bool:
    text = str(found)
    if text == entry or text.upper() == entry.upper():
        return True
    try:
        return float(text) == float(entry)
    except ValueError:
        return False


def resolve_perturbations(entries: Iterable[str] | str,
                          modality: str) -> dict[str, list[str]]:
    """
    {entry: [JCP2022 ids]} for perturbation identifiers of one modality
    ("compound", "crispr" or "orf").

    JCP2022 ids are kept as they are. Anything else is looked up with
    broad_babel in its standard_key (InChIKey or gene symbol), broad_sample,
    NCBI_Gene_ID and JCP2022 columns, keeping only rows of `modality`; gene
    symbols and InChIKeys also match in upper case. An entry may resolve to
    several ids. Raises ValueError naming the entries that resolve to none.
    """
    if isinstance(entries, str):
        entries = [entries]
    entries = list(dict.fromkeys(str(e).strip() for e in entries))
    if not entries:
        raise ValueError("No perturbations given.")

    resolved: dict[str, set[str]] = {e: set() for e in entries}
    pending = [e for e in entries if not _JCP2022_ID.fullmatch(e)]
    for e in entries:
        if e not in pending:
            resolved[e].add(e)

    if pending:
        try:
            from broad_babel.query import run_query
        except ImportError as error:
            raise ImportError(
                "Translating identifiers other than JCP2022 ids needs "
                "broad_babel (pip install broad-babel).") from error
        queries = tuple(sorted(set(pending) | {e.upper() for e in pending}))
        for column in _BABEL_COLUMNS:
            rows = run_query(queries, input_column=column,
                             output_columns=f"{column},JCP2022,plate_type")
            for found, jcp, plate_type in rows:
                if not jcp or str(plate_type).lower() != modality.lower():
                    continue
                for entry in pending:
                    if _same_identifier(found, entry):
                        resolved[entry].add(jcp)

    unresolved = [e for e, ids in resolved.items() if not ids]
    if unresolved:
        raise ValueError(
            f"No {modality} JCP2022 id found for: {unresolved}. broad_babel "
            f"translates JCP2022 ids, InChIKeys and gene symbols, Broad sample "
            f"ids and NCBI gene ids.")
    return {e: sorted(ids) for e, ids in resolved.items()}


def _check_selection(plates: pd.DataFrame, plate_types: Iterable[str] | None,
                     exclude: list) -> None:
    """Raise ValueError for plate types or sources the plate table lacks."""
    known_types = set(plates["Metadata_PlateType"].astype(str).str.upper())
    known_sources = set(plates["Metadata_Source"])
    if plate_types is not None:
        unknown = {t.upper() for t in plate_types} - known_types
        if unknown:
            raise ValueError(f"Unknown plate type(s) {sorted(unknown)}; the plate "
                             f"table has {sorted(known_types)}.")
    for item in exclude:
        source, excluded_type = (item, None) if isinstance(item, str) else tuple(item)
        if source not in known_sources:
            raise ValueError(
                f"Unknown source {source!r} in exclude. Give source names or "
                f"(source, plate type) pairs, e.g. "
                f"[\"source_7\", (\"source_9\", \"TARGET2\")].")
        if excluded_type is not None and excluded_type.upper() not in known_types:
            raise ValueError(f"Unknown plate type {excluded_type!r} in exclude.")


def _select_plates(plates: pd.DataFrame, plate_types: Iterable[str] | None,
                   exclude: list) -> pd.DataFrame:
    """Rows of the plate table with one of `plate_types` (None: any type) and
    not excluded. `exclude` holds source names, or (source, plate type)
    pairs that leave out only that source's plates of that type."""
    plate_type = plates["Metadata_PlateType"].astype(str).str.upper()
    keep = np.ones(len(plates), dtype=bool)
    if plate_types is not None:
        keep &= plate_type.isin({t.upper() for t in plate_types}).to_numpy()
    for item in exclude:
        source, excluded_type = (item, None) if isinstance(item, str) else tuple(item)
        hit = (plates["Metadata_Source"] == source).to_numpy()
        if excluded_type is not None:
            hit = hit & (plate_type == excluded_type.upper()).to_numpy()
        keep &= ~hit
    return plates[keep]


def _drop_incomplete_sites(site_table: pd.DataFrame,
                           channels: list[str]) -> pd.DataFrame:
    incomplete = site_table[[url_column(c) for c in channels]].isna().any(axis=1)
    if incomplete.any():
        by_source = site_table.loc[incomplete, "Metadata_Source"].value_counts()
        warnings.warn(
            f"Dropping {int(incomplete.sum())} of {len(site_table)} site(s) "
            f"without an image for every requested channel, by source: "
            f"{by_source.to_dict()}", stacklevel=3)
        site_table = site_table[~incomplete.to_numpy()].reset_index(drop=True)
    return site_table


def _site_table_from_metadata(perturbations, plate_types, exclude,
                              negative_controls, controls, channels):
    """
    The site table of the wells of `perturbations` (None: every well) on the
    plates of `plate_types` that are not excluded, plus negative-control wells:

      controls="plates"   those on the plates carrying a selected perturbation
      controls="batches"  those on every plate, of any type, in the batches of
                          those plates (exclusions still apply)
      controls="all"      those on every selected plate
      controls="none"     none
    """
    if controls not in CONTROL_SCOPES:
        raise ValueError(f"controls must be one of {CONTROL_SCOPES}, not {controls!r}")
    exclude = [exclude] if isinstance(exclude, str) else list(exclude)
    plates = load_metadata("plate")
    _check_selection(plates, plate_types, exclude)
    selected_plates = _select_plates(plates, plate_types, exclude)
    if selected_plates.empty:
        raise ValueError("No plates of the requested types remain after the exclusions.")

    well = load_metadata("well")
    plate_keys = ["Metadata_Source", "Metadata_Plate"]
    on_selected = well.merge(selected_plates[plate_keys], on=plate_keys, how="inner")

    if perturbations is None:
        treated = on_selected.dropna(subset=["Metadata_JCP2022"])
    else:
        treated = on_selected[on_selected["Metadata_JCP2022"].isin(perturbations)]
        absent = sorted(set(perturbations) - set(treated["Metadata_JCP2022"]))
        if len(absent) == len(set(perturbations)):
            raise ValueError("None of the requested perturbations is on the selected plates.")
        if absent:
            warnings.warn(f"{len(absent)} perturbation(s) not on the selected "
                          f"plates: {absent}", stacklevel=3)

    is_control = well["Metadata_JCP2022"].isin(list(negative_controls))
    if controls == "plates":
        carrying = treated[plate_keys].drop_duplicates()
        control_wells = well[is_control].merge(carrying, on=plate_keys, how="inner")
    elif controls == "batches":
        batch_keys = ["Metadata_Source", "Metadata_Batch"]
        batches = (selected_plates.merge(treated[plate_keys].drop_duplicates(),
                                         on=plate_keys, how="inner")
                   [batch_keys].drop_duplicates())
        in_batches = _select_plates(plates.merge(batches, on=batch_keys), None, exclude)
        control_wells = well[is_control].merge(in_batches[plate_keys], on=plate_keys,
                                               how="inner")
    elif controls == "all":
        control_wells = on_selected[on_selected["Metadata_JCP2022"].isin(list(negative_controls))]
    else:
        control_wells = well.iloc[0:0]

    wells = pd.concat([treated, control_wells], ignore_index=True)
    return _drop_incomplete_sites(site_table_for_wells(wells, channels), channels)


# Datasets

class JUMPCellPaintingDataset(Dataset):
    """
    Streaming dataset over the sites of a site table.

    Args:
        site_table: one row per site, with SITE_KEYS, Metadata_JCP2022 and a
            URL_Orig<channel> column for each channel. Other columns are kept
            in self.site_table but not returned by __getitem__. Every site
            must have an image for every channel.
        channels: channel names, in the order of the image's channel axis
            (default FLUORESCENT_CHANNELS).

    Each item is a dict with "image" (C, H, W) and the site's Metadata_Source,
    Metadata_Batch, Metadata_Plate, Metadata_Well, Metadata_Site and
    Metadata_JCP2022.
    """

    def __init__(self, site_table: pd.DataFrame,
                 channels: Iterable[str] | None = None):
        self.channels = resolve_channels(channels)
        self.url_columns = [url_column(c) for c in self.channels]

        required = SITE_KEYS + ["Metadata_JCP2022"] + self.url_columns
        missing = [c for c in required if c not in site_table.columns]
        if missing:
            raise ValueError(f"The site table has no column {', '.join(missing)}.")
        incomplete = site_table[self.url_columns].isna().any(axis=1)
        if incomplete.any():
            by_source = site_table.loc[incomplete, "Metadata_Source"].value_counts()
            raise ValueError(
                f"{int(incomplete.sum())} site(s) have no image for some of "
                f"{self.channels}, by source: {by_source.to_dict()}. Drop them "
                f"or request fewer channels.")
        self.site_table = site_table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.site_table)

    def __getitem__(self, i: int) -> dict:
        from jump_portrait.s3 import get_image_from_s3uri

        row = self.site_table.iloc[i]
        image = np.stack([get_image_from_s3uri(row[c]) for c in self.url_columns], axis=0)
        return {
            "image":            image,
            "Metadata_Source":  row["Metadata_Source"],
            "Metadata_Batch":   row["Metadata_Batch"],
            "Metadata_Plate":   row["Metadata_Plate"],
            "Metadata_Well":    row["Metadata_Well"],
            "Metadata_Site":    row["Metadata_Site"],
            "Metadata_JCP2022": row["Metadata_JCP2022"],
        }


class JUMPTarget2Dataset(JUMPCellPaintingDataset):
    """
    Every well of the Target-2 plates: the Target-2 compounds and their DMSO
    wells.

    Args:
        exclude: source names, or (source, plate type) pairs, to leave out
            (default source_9, whose Target-2 plates have a known problem;
            see https://github.com/broadinstitute/jump_hub/issues/114).
        controls: "plates" (default) adds nothing beyond the plates' own DMSO
            wells; "batches" adds the DMSO wells of every other plate in the
            Target-2 plates' batches, which gives normalizers fitted per batch
            more controls.
        channels: as for JUMPCellPaintingDataset.
    """

    def __init__(self, exclude: Iterable = ("source_9",), controls: str = "plates",
                 channels: Iterable[str] | None = None):
        channels = resolve_channels(channels)
        site_table = _site_table_from_metadata(
            None, ("TARGET2",), exclude, control_ids("negcon", "compound"),
            controls, channels)
        super().__init__(site_table, channels)


class JUMPCompoundDataset(JUMPCellPaintingDataset):
    """
    The wells of the given compounds on compound-screen plates, and the
    negative-control (DMSO) wells of the plates they are on.

    Args:
        compounds: identifiers broad_babel translates (see
            resolve_perturbations); the resolution is kept in
            self.perturbations.
        exclude: source names, or (source, plate type) pairs, to leave out.
            Default: source_7, whose COMPOUND plates are dosed at 0.625 uM
            rather than 10 uM, and source_9's Target-2 plates (see
            JUMPTarget2Dataset).
        plate_types: plate types to use (default COMPOUND_PLATE_TYPES, every
            type except ORF and CRISPR); None for any type.
        controls: which negative controls to add, "plates" (default),
            "batches", "all" or "none"; see _site_table_from_metadata.
        negative_controls: JCP2022 ids of the negative controls (default:
            the compound negcons of the perturbation_control table, DMSO).
        channels: as for JUMPCellPaintingDataset.
    """

    def __init__(self, compounds: Iterable[str] | str,
                 exclude: Iterable = ("source_7", ("source_9", "TARGET2")),
                 plate_types: Iterable[str] | None = COMPOUND_PLATE_TYPES,
                 controls: str = "plates",
                 negative_controls: Iterable[str] | None = None,
                 channels: Iterable[str] | None = None):
        channels = resolve_channels(channels)
        if negative_controls is None:
            negative_controls = control_ids("negcon", "compound")
        self.perturbations = resolve_perturbations(compounds, "compound")
        ids = sorted({j for js in self.perturbations.values() for j in js})
        site_table = _site_table_from_metadata(
            ids, plate_types, exclude, tuple(negative_controls), controls, channels)
        super().__init__(site_table, channels)


class JUMPPositiveControlDataset(JUMPCompoundDataset):
    """JUMPCompoundDataset for the compound positive controls of the
    perturbation_control table. Takes the same keyword arguments except
    `compounds`."""

    def __init__(self, **kwargs):
        super().__init__(control_ids("poscon", "compound"), **kwargs)


class JUMPCRISPRDataset(JUMPCellPaintingDataset):
    """
    The wells of the CRISPR guides against the given genes, and the
    negative-control wells of the plates they are on (the CRISPR negcons of
    the perturbation_control table: non-targeting and no-guide). All CRISPR
    plates are in source_13.

    Args:
        genes: identifiers broad_babel translates (see resolve_perturbations),
            e.g. gene symbols or NCBI gene ids; the resolution is kept in
            self.perturbations.
        exclude: source names, or (source, plate type) pairs (default none).
        controls: which negative controls to add, "plates" (default),
            "batches", "all" or "none"; see _site_table_from_metadata.
        negative_controls: JCP2022 ids of the negative controls (default:
            the CRISPR negcons of the perturbation_control table).
        channels: as for JUMPCellPaintingDataset.
    """

    def __init__(self, genes: Iterable[str] | str, exclude: Iterable = (),
                 controls: str = "plates",
                 negative_controls: Iterable[str] | None = None,
                 channels: Iterable[str] | None = None):
        channels = resolve_channels(channels)
        if negative_controls is None:
            negative_controls = control_ids("negcon", "crispr")
        self.perturbations = resolve_perturbations(genes, "crispr")
        ids = sorted({j for js in self.perturbations.values() for j in js})
        site_table = _site_table_from_metadata(
            ids, ("CRISPR",), exclude, tuple(negative_controls), controls, channels)
        super().__init__(site_table, channels)
