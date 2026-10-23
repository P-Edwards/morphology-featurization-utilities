"""
Retrieval scoring of well-level JUMP profiles with copairs: metadata joins,
plate and source filters, per-block normalization, and Design B.

Design B (Kalinin et al. 2025, Nat Commun 16:5181): phenotypic activity, i.e.
retrieval of a compound's replicate wells against DMSO wells. In strict mode
each compound gets R replicate wells on R distinct plates and is ranked
against the DMSO wells of those plates, repeated over random draws.

Functions that take `args` expect an argparse namespace carrying the options
added by add_preprocess_arguments and the design-specific arguments of the
calling script.

copairs does not build a distance matrix; it materializes every pair and sorts
an array of length 2 x n_pairs, which peaks at about BYTES_PER_PAIR bytes per
pair (traced against copairs 0.5.4). Designs whose pair count would exceed the
memory budget are skipped rather than attempted.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from copairs import map as copairs_map
from copairs.matching import assign_reference_index

from .normalizers import MADRobustize, TypicalVariationNormalization

# Same value as datasets.jump_cell_painting.DMSO_JCP2022, which cannot be
# imported here without torch.
DMSO_JCP2022 = "JCP2022_033924"

WELL_KEYS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
REFERENCE_COL = "Metadata_reference_index"

BYTES_PER_PAIR = 84
GiB = 1024 ** 3

# --n-components auto: largest TVN rank with this many controls per dimension
# in the 10th-percentile batch.
SAMPLES_PER_DIMENSION = 10


# Feature blocks. Extractors prefix their columns with "<family>__"; every
# cp_* family belongs to one CellProfiler block.

def block_of(column: str) -> str:
    family = column.split("__", 1)[0]
    return "cellprofiler" if family.startswith("cp_") else family


def split_blocks(feature_cols: list[str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for column in feature_cols:
        blocks.setdefault(block_of(column), []).append(column)
    return blocks


# Degeneracy guards. A constant or empty block makes every cosine equal, and
# copairs then reports mAP 1.0 and 100% retrieval.

def screen_columns(values: np.ndarray, columns: list[str]):
    """(indices of usable columns, names of unusable ones). A column is
    unusable if it has no finite value or is constant."""
    finite = np.isfinite(values)
    all_missing = ~finite.any(axis=0)

    constant = np.zeros(values.shape[1], dtype=bool)
    for index in np.flatnonzero(~all_missing):
        column = values[finite[:, index], index]
        constant[index] = column.max() == column.min()

    unusable = all_missing | constant
    return np.flatnonzero(~unusable), [columns[i] for i in np.flatnonzero(unusable)]


def direction_diagnostics(values: np.ndarray, max_rows: int = 2000,
                          decimals: int = 6) -> tuple[float, float]:
    """(fraction of rows with a distinct direction, largest tied fraction)."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return 0.0, 1.0
    if len(array) > max_rows:
        index = np.random.default_rng(0).choice(len(array), max_rows, replace=False)
        array = array[index]
    norms = np.linalg.norm(array, axis=1)
    usable = norms > 0
    if not usable.any():
        return 0.0, 1.0
    unit = np.round(array[usable] / norms[usable, None], decimals)
    _, counts = np.unique(unit, axis=0, return_counts=True)
    return len(counts) / len(unit), counts.max() / len(unit)


def is_degenerate(values: np.ndarray, min_distinct_fraction: float = 0.5,
                  max_tied_fraction: float = 0.10) -> str | None:
    """Reason `values` cannot be meaningfully scored, or None."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] == 0:
        return "fewer than two rows or no columns"
    if not np.isfinite(array).any():
        return "no finite values"
    distinct, largest_tie = direction_diagnostics(array)
    if distinct < min_distinct_fraction:
        return (f"only {100 * distinct:.1f}% of rows have a distinct direction "
                f"(cosine retrieval cannot rank tied rows)")
    if largest_tie > max_tied_fraction:
        return f"{100 * largest_tie:.1f}% of rows share one direction"
    return None


# JUMP metadata

def load_jump_table(name_candidates, cache_dir: Path, log) -> pd.DataFrame | None:
    """
    First JUMP metadata table that loads, cached as <cache_dir>/jump_<name>.csv.
    jump_portrait's table names have changed between releases, so several
    names can be tried.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in name_candidates:
        cached = cache_dir / f"jump_{name}.csv"
        if cached.exists():
            log(f"Using cached JUMP '{name}' table: {cached}")
            return pd.read_csv(cached)
    from jump_portrait.fetch import get_table
    for name in name_candidates:
        try:
            frame = pd.read_csv(get_table(name))
        except Exception:                                       # noqa: BLE001
            continue
        cached = cache_dir / f"jump_{name}.csv"
        frame.to_csv(cached, index=False)
        log(f"Fetched and cached JUMP '{name}' table -> {cached}")
        return frame
    return None


def attach_metadata(profile: pd.DataFrame, cache_dir: Path, log) -> pd.DataFrame:
    """Join Metadata_JCP2022 from the well table, and Metadata_Batch and
    Metadata_PlateType from the plate table."""
    well = load_jump_table(("well",), cache_dir, log)
    if well is None:
        raise SystemExit("Could not load the JUMP 'well' table; compound ids "
                         "are required. Supply a local copy as "
                         "<cache>/jump_well.csv.")
    lookup = (well.loc[well["Metadata_Plate"].isin(set(profile["Metadata_Plate"])),
                       WELL_KEYS + ["Metadata_JCP2022"]]
              .drop_duplicates(WELL_KEYS)
              .reset_index(drop=True))
    profile = profile.merge(lookup, on=WELL_KEYS, how="left")

    plates = load_jump_table(("plate", "plates"), cache_dir, log)
    if plates is None or "Metadata_Batch" not in (plates.columns if plates is not None else []):
        log("WARNING: no plate table / no Metadata_Batch column. Falling back "
            "to Metadata_Plate as the batch key, so cross-batch matching "
            "becomes cross-plate matching.")
        profile["Metadata_Batch"] = profile["Metadata_Plate"].astype(str)
        return profile

    on = ["Metadata_Source", "Metadata_Plate"]
    if "Metadata_Source" not in plates.columns:
        on = ["Metadata_Plate"]
    extra = [c for c in ("Metadata_Batch", "Metadata_PlateType")
             if c in plates.columns]
    batch_lookup = (plates[on + extra]
                    .drop_duplicates(on)
                    .reset_index(drop=True))
    return profile.merge(batch_lookup, on=on, how="left")


# Normalization

def center_by_group(X: np.ndarray, control_mask: np.ndarray,
                    groups: np.ndarray, level: str, log) -> np.ndarray:
    """
    Subtract each group's negative-control mean.

    Ando, McLean & Berndl (2017) fit TVN per experiment, which places the
    origin at that experiment's control mean; fitting one TVN across the whole
    screen would instead centre on a global mean. Kraus et al. (2024) Table 2:
    centring by plate or by experiment (.449/.455 recall on CORUM) is the
    largest single gain, while global centring alone (.126) adds nothing.
    """
    if level == "none":
        return X
    out = X.copy()
    missing = 0
    for group in np.unique(groups):
        in_group = groups == group
        controls_here = in_group & control_mask
        if not controls_here.any():
            missing += 1
            continue
        out[in_group] -= X[controls_here].mean(axis=0)
    if missing:
        log(f"    {missing} {level}(s) had no controls and were left uncentred")
    return out


def choose_rank(profile: pd.DataFrame, control_mask: np.ndarray,
                n_features: int, log) -> int:
    """
    Largest TVN rank the per-batch control counts support.

    TVN's CORAL step estimates a k x k covariance per batch from that batch's
    controls, so k is limited by the smallest batches; beyond that the
    covariance is singular and its null directions are amplified by
    1/sqrt(epsilon). Uses the 10th-percentile batch and SAMPLES_PER_DIMENSION
    controls per dimension.
    """
    counts = profile.loc[control_mask, "Metadata_Batch"].value_counts()
    floor = int(np.percentile(counts.values, 10))
    rank = max(2, min(n_features, floor // SAMPLES_PER_DIMENSION))
    log(f"Controls per batch: min {counts.min()}, p10 {floor}, "
        f"median {int(counts.median())}, max {counts.max()} "
        f"across {len(counts)} batch(es).")
    log(f"  rank auto -> {rank} (p10 / {SAMPLES_PER_DIMENSION} controls per dimension)")
    thin = int((counts < SAMPLES_PER_DIMENSION * rank).sum())
    if thin:
        log(f"  WARNING: {thin} batch(es) have fewer than "
            f"{SAMPLES_PER_DIMENSION}x{rank} controls.")
    return rank


def _resolve_rank(requested, profile, control_mask, n_features, log) -> int:
    if str(requested) == "auto":
        return choose_rank(profile, control_mask, n_features, log)
    return int(requested)


def normalize_blocks(profile, blocks, control_mask, plates, args, report):
    """
    Normalize each feature block separately.

    MAD is fitted per plate. TVN centres on the control mean per
    --center-level group and fits its CORAL step per batch, since a single
    plate's controls cannot support a useful covariance rank. With
    --normalizer none the features are only screened.

    Returns (normalized, usable_columns, degenerate, block_report).
    """
    normalized: dict[str, np.ndarray] = {}
    usable_columns: dict[str, list[str]] = {}
    degenerate: dict[str, str] = {}
    block_report: dict[str, dict] = {}
    n_controls = int(control_mask.sum())

    level = args.center_level if args.normalizer == "tvn" else "none"
    centring = level != "none"
    if centring:
        groups = (profile["Metadata_Batch"].to_numpy() if level == "batch"
                  else plates)
        report.log(f"Centring at the negative-control mean per {level}.")

    for name in sorted(blocks):
        cols = blocks[name]
        raw = profile[cols].to_numpy(dtype=float)
        missing_fraction = float(np.mean(~np.isfinite(raw)))
        usable_idx, dropped = screen_columns(raw, cols)

        if dropped:
            report.log(f"  [{name}] dropping {len(dropped)}/{len(cols)} column(s) "
                       f"with no variance or no finite values")
        if usable_idx.size == 0:
            degenerate[name] = (f"all {len(cols)} columns constant or non-finite "
                                f"({100 * missing_fraction:.0f}% non-finite)")
            report.log(f"{args.normalizer.upper()}[{name}]: SKIPPED -- {degenerate[name]}")
            continue

        X = np.nan_to_num(raw[:, usable_idx], nan=0.0, posinf=0.0, neginf=0.0)
        if centring:
            X = center_by_group(X, control_mask, groups, level, report.log)

        if args.normalizer == "mad":
            normalizer = MADRobustize()
            normalizer.fit(X, control_mask=control_mask, batch_labels=plates)
            Xn = normalizer.transform(X, batch_labels=plates)
        elif args.normalizer == "none":
            Xn = X
        else:
            requested = args.block_components.get(name, args.n_components)
            requested = _resolve_rank(requested, profile, control_mask,
                                      X.shape[1], report.log)
            k = min(requested, X.shape[1], max(1, n_controls))
            normalizer = TypicalVariationNormalization(n_components=k)
            normalizer.fit(X, control_mask=control_mask,
                           batch_labels=args.batch_labels)
            Xn = normalizer.transform(X, batch_labels=args.batch_labels)

        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        reason = is_degenerate(Xn)
        if reason is not None:
            degenerate[name] = f"after {args.normalizer}: {reason}"
            report.log(f"{args.normalizer.upper()}[{name}]: SKIPPED -- {degenerate[name]}")
            continue

        distinct, largest_tie = direction_diagnostics(Xn)
        normalized[name] = Xn
        usable_columns[name] = [cols[i] for i in usable_idx]
        block_report[name] = {
            "in_features": len(cols),
            "usable_features": int(usable_idx.size),
            "non_finite_fraction": missing_fraction,
            "out_components": int(Xn.shape[1]),
            "distinct_direction_fraction": round(distinct, 4),
            "largest_tied_fraction": round(largest_tie, 4),
        }
        report.log(f"{args.normalizer.upper()}[{name}]: {usable_idx.size}/{len(cols)} "
                   f"usable features -> {Xn.shape[1]} components "
                   f"({100 * missing_fraction:.1f}% non-finite, n_controls={n_controls})")

    return normalized, usable_columns, degenerate, block_report


def normalize_matrix(X, control_mask, plates, args, k, label, report):
    """Normalize one raw matrix as normalize_blocks does a block, with TVN
    rank k. Returns None if the result is degenerate."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if args.center_level != "none" and args.normalizer == "tvn":
        groups = plates if args.center_level == "plate" else args.batch_labels
        X = center_by_group(X, control_mask, groups, args.center_level, report.log)

    if args.normalizer == "none":
        Xn = X
    else:
        if args.normalizer == "mad":
            normalizer = MADRobustize()
            groups = plates
        else:
            rank = min(k, X.shape[1], max(1, int(control_mask.sum())))
            normalizer = TypicalVariationNormalization(n_components=rank)
            groups = args.batch_labels
        normalizer.fit(X, control_mask=control_mask, batch_labels=groups)
        Xn = np.nan_to_num(normalizer.transform(X, batch_labels=groups),
                           nan=0.0, posinf=0.0, neginf=0.0)

    reason = is_degenerate(Xn)
    if reason is not None:
        report.log(f"  [{label}] SKIPPED -- {reason}")
        return None
    report.log(f"  [{label}] {X.shape[1]} raw feature(s) -> {Xn.shape[1]} component(s)")
    return Xn


# Concatenation and its shuffle control

def shuffle_within_strata(X, strata, rng):
    """
    Permute rows within each (plate x control-status) stratum.

    This keeps plate structure and any systematic treated-vs-control
    difference, including row norms, while destroying which well each row
    belongs to. A concatenation that still scores well after its second block
    is shuffled is not gaining compound-specific information from that block.
    """
    out = X.copy()
    for stratum in np.unique(strata):
        index = np.flatnonzero(strata == stratum)
        out[index] = X[rng.permutation(index)]
    return out


def build_strata(profile, control_mask):
    return np.array([f"{p}|{int(c)}" for p, c in
                     zip(profile["Metadata_Plate"].to_numpy(), control_mask)])


def build_feature_sets(profile, normalized, usable_columns, control_mask,
                       plates, args, report):
    """
    The per-block feature sets plus, with two or more blocks, their
    concatenation (and its shuffle control if requested).

    --concat-order joint concatenates the raw features and normalizes once, so
    the whitening puts all blocks on one scale and can remove nuisance shared
    across blocks. --concat-order per-block stacks the separately normalized
    blocks. --joint-components 0 means full rank: at a truncated joint rank,
    wide blocks dominate the leading components.
    """
    sets = dict(normalized)
    names = sorted(normalized)
    if len(names) < 2:
        return sets

    if args.concat_order == "per-block":
        report.log("\nConcatenation: per-block normalization, then stacked.")
        sets["concatenated"] = np.hstack([normalized[n] for n in names])
        if args.shuffle_control == "on":
            rng = np.random.default_rng(args.seed)
            strata = build_strata(profile, control_mask)
            parts = [normalized[names[0]]] + [
                shuffle_within_strata(normalized[n], strata, rng) for n in names[1:]]
            sets["concatenated_shuffled"] = np.hstack(parts)
    else:
        report.log("\nConcatenation: raw features stacked, then normalized once.")
        raw = {n: profile[usable_columns[n]].to_numpy(dtype=float) for n in names}
        k = args.joint_components or sum(v.shape[1] for v in raw.values())
        joint = normalize_matrix(np.hstack([raw[n] for n in names]), control_mask,
                                 plates, args, k, "concatenated", report)
        if joint is not None:
            sets["concatenated"] = joint
        if args.shuffle_control == "on":
            rng = np.random.default_rng(args.seed)
            strata = build_strata(profile, control_mask)
            report.log(f"  shuffle control: rows of {', '.join(names[1:])} "
                       f"permuted within (plate x control-status); "
                       f"{names[0]} left intact")
            parts = [raw[names[0]]] + [
                shuffle_within_strata(raw[n], strata, rng) for n in names[1:]]
            shuffled = normalize_matrix(np.hstack(parts), control_mask, plates,
                                        args, k, "concatenated_shuffled", report)
            if shuffled is not None:
                sets["concatenated_shuffled"] = shuffled
    return sets


def report_budget(n_pairs: int, label: str, budget_gib: float, report) -> bool:
    """Log the projected copairs peak; False if it exceeds the budget."""
    gib = n_pairs * BYTES_PER_PAIR / GiB
    report.log(f"  [{label}] {n_pairs:,} pair(s) -> ~{gib:,.1f} GiB peak "
               f"(budget {budget_gib:,.0f} GiB)")
    if gib > budget_gib:
        report.log(f"  [{label}] SKIPPED -- projected peak exceeds the budget. "
                   f"Lower --plates-per-draw, or raise --memory-budget-gib if "
                   f"the node has the memory.")
        return False
    return True


# Preprocessing

def add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that determine the normalized feature sets."""
    group = parser.add_argument_group("preprocessing")
    group.add_argument("--featurization", required=True, type=Path,
                       help="Featurization run directory (contains "
                            "well_profiles.csv), from jump_featurization or "
                            "jump_consortium_profiles.")
    group.add_argument("--cache", type=Path, required=True,
                       help="Persistent shared directory for the JUMP metadata "
                            "tables and the copairs null distributions.")
    group.add_argument("--normalizer", default="tvn", choices=["tvn", "mad", "none"],
                       help="'none' for profiles that are already batch "
                            "corrected, such as the consortium's Harmony table.")
    group.add_argument("--center-level", default="plate",
                       choices=["plate", "batch", "none"],
                       help="Group whose control mean is subtracted before TVN. "
                            "No effect with MAD, which centres per plate.")
    group.add_argument("--n-components", default="40",
                       help="TVN rank for blocks not named in --block-components; "
                            "an integer or 'auto' (see choose_rank).")
    group.add_argument("--block-components", default="tda=40,autoencoder=40",
                       help="Per-block TVN ranks, 'name=rank,...'; rank may be 'auto'.")
    group.add_argument("--concat-order", default="joint",
                       choices=["joint", "per-block"],
                       help="Build the concatenated set from raw features "
                            "normalized once (joint), or by stacking the "
                            "normalized blocks (per-block).")
    group.add_argument("--joint-components", type=int, default=0,
                       help="TVN rank of the joint concatenation; 0 = full rank.")
    group.add_argument("--shuffle-control", default="on", choices=["on", "off"],
                       help="Also score the concatenation with the second and "
                            "later blocks shuffled within (plate x control-status).")
    group.add_argument("--plate-types", default="",
                       help="Comma-separated Metadata_PlateType values of the "
                            "plates to score, e.g. TARGET2. Empty scores all. "
                            "Plate types carry different doses (Target-2 5 uM, "
                            "production compound plates 10 uM, source_7 "
                            "0.625 uM).")
    group.add_argument("--fit-controls", default="all",
                       help="Control wells the normalizer is fitted on besides "
                            "those of the scored plates. all: those on every "
                            "other plate; scored: none; or comma-separated "
                            "plate types, e.g. COMPOUND,DMSO. They are dropped "
                            "after normalization. Matters for TVN, whose "
                            "centring, PCA and per-batch CORAL use every "
                            "control of a batch; MAD is fitted per plate and "
                            "is unaffected.")
    group.add_argument("--exclude-sources", default="source_9",
                       help="Comma-separated sources dropped before "
                            "normalization, so their controls do not enter "
                            "the TVN fit. Empty keeps all.")
    group.add_argument("--intersect-with", type=Path, default=None,
                       help="Keep only wells present in this table (a "
                            "well_profiles.csv, parquet or csv with "
                            "Metadata_Source/Plate/Well), e.g. the consortium "
                            "profiles, whose QC removed some plates.")


def parse_block_components(text: str) -> dict:
    overrides = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, rank = item.partition("=")
        rank = rank.strip()
        overrides[name.strip()] = rank if rank == "auto" else int(rank)
    return overrides


def _parse_list(text: str) -> list[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_and_prepare(args, report):
    """
    Read a featurization's well profiles, attach metadata, apply the filters
    and normalize every feature block.

    Returns (profile, feature_sets, info). `profile` holds the metadata of the
    wells to score, row-aligned with every array in `feature_sets`. Control
    wells on plates outside --plate-types take part in the normalization fit
    as --fit-controls says, and are not returned.
    """
    if isinstance(args.block_components, str):
        args.block_components = parse_block_components(args.block_components)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    well_profiles = Path(args.featurization) / "well_profiles.csv"
    if not well_profiles.exists():
        raise SystemExit(f"No well_profiles.csv in {args.featurization}. Point "
                         f"--featurization at a completed featurization run.")
    report.log(f"Well profiles: {well_profiles}")
    selection = Path(args.featurization) / "selection.json"
    if selection.exists():
        import json
        sel = json.loads(selection.read_text())
        report.log(f"Featurization selection: mode={sel.get('selection_mode')}, "
                   f"{len(sel.get('selected', []))} compound(s), seed={sel.get('seed')}")

    profile = pd.read_csv(well_profiles)
    feature_cols = [c for c in profile.columns if c not in WELL_KEYS]
    blocks = split_blocks(feature_cols)
    report.log(f"Loaded {profile.shape[0]:,} well(s) x {len(feature_cols):,} feature(s)")
    report.log("Feature blocks: " + ", ".join(
        f"{name} ({len(cols)})" for name, cols in sorted(blocks.items())))

    profile = attach_metadata(profile, cache_dir, report.log)

    unlabelled = profile["Metadata_JCP2022"].isna()
    if unlabelled.any():
        report.log(f"Dropping {int(unlabelled.sum())} well(s) with no "
                   f"Metadata_JCP2022 after the well-table join.")
        profile = profile[~unlabelled].reset_index(drop=True)

    missing_batch = profile["Metadata_Batch"].isna()
    if missing_batch.any():
        report.log(f"Dropping {int(missing_batch.sum())} well(s) with no "
                   f"Metadata_Batch after the plate-table join.")
        profile = profile[~missing_batch].reset_index(drop=True)

    if "Metadata_PlateType" in profile.columns and "Metadata_Source" in profile.columns:
        counts = pd.crosstab(profile["Metadata_Source"], profile["Metadata_PlateType"])
        report.log("\nWells by source and plate type:")
        report.log(counts.to_string())

    # Rows marked _fit_only are control wells on plates that are not scored;
    # they take part in the normalization fit and are dropped afterwards.
    profile["_fit_only"] = False
    keep_types = [x.lower() for x in _parse_list(args.plate_types)]
    if keep_types:
        if "Metadata_PlateType" not in profile.columns:
            report.log("--plate-types given but the plate table has no "
                       "Metadata_PlateType column; nothing filtered.")
        else:
            before = len(profile)
            scored = (profile["Metadata_PlateType"].astype(str).str.lower()
                      .isin(keep_types)).to_numpy()
            report.log(f"\n--plate-types {', '.join(keep_types)}: scoring "
                       f"{int(scored.sum()):,}/{before:,} well(s) "
                       f"({100 * scored.sum() / max(before, 1):.1f}%), "
                       f"{profile.loc[scored, 'Metadata_Plate'].nunique():,} plate(s).")
            keep = scored.copy()
            fit_types = args.fit_controls.strip().lower()
            if fit_types != "scored":
                extra = ~scored & (profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()
                if fit_types != "all":
                    extra &= (profile["Metadata_PlateType"].astype(str).str.lower()
                              .isin(_parse_list(fit_types))).to_numpy()
                keep |= extra
                profile["_fit_only"] = extra
                report.log(f"  {int(extra.sum()):,} control well(s) on "
                           f"{profile.loc[extra, 'Metadata_Plate'].nunique():,} "
                           f"other plate(s) are used for the normalization fit "
                           f"only.")
            profile = profile[keep].reset_index(drop=True)
            if not (~profile["_fit_only"]).any():
                raise SystemExit("No wells remain after the plate-type filter.")

    excluded = _parse_list(args.exclude_sources)
    if excluded:
        if "Metadata_Source" not in profile.columns:
            report.log("--exclude-sources given but there is no Metadata_Source "
                       "column; nothing dropped.")
        else:
            drop = profile["Metadata_Source"].isin(excluded).to_numpy()
            report.log(f"Excluding source(s) {', '.join(excluded)}: dropping "
                       f"{int(drop.sum()):,} well(s) before normalization.")
            missing = set(excluded) - set(profile["Metadata_Source"])
            if missing:
                report.log(f"  NOTE: {', '.join(sorted(missing))} not present "
                           f"in this run.")
            profile = profile[~drop].reset_index(drop=True)
            if profile.empty:
                raise SystemExit("No wells remain after excluding sources.")

    if args.intersect_with:
        ref_path = Path(args.intersect_with)
        if ref_path.is_dir():
            ref_path = ref_path / "well_profiles.csv"
        if not ref_path.exists():
            raise SystemExit(f"--intersect-with: {ref_path} not found.")
        if ref_path.suffix == ".parquet":
            import pyarrow.parquet as pq
            ref_keys = pq.read_table(ref_path, columns=WELL_KEYS).to_pandas()
        else:
            ref_keys = pd.read_csv(ref_path, usecols=WELL_KEYS)
        ref_keys = ref_keys.drop_duplicates()
        before = len(profile)
        profile = profile.merge(ref_keys, on=WELL_KEYS, how="inner"
                                ).reset_index(drop=True)
        report.log(f"--intersect-with {ref_path}: {len(profile):,}/{before:,} "
                   f"well(s) retained ({100 * len(profile) / max(before, 1):.1f}%).")
        if profile.empty:
            raise SystemExit("No wells remain after intersection.")

    control_mask = (profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()
    with_control_rows = set(profile.loc[control_mask, "Metadata_Plate"])
    uncontrolled = sorted(set(profile["Metadata_Plate"]) - with_control_rows)
    if uncontrolled:
        keep = profile["Metadata_Plate"].isin(with_control_rows).to_numpy()
        report.log(f"Dropping {len(uncontrolled)} plate(s) with no control wells "
                   f"({int((~keep).sum())} well(s)); the normalizer cannot fit "
                   f"them. First few: {uncontrolled[:5]}")
        profile = profile[keep].reset_index(drop=True)
        control_mask = (profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()

    if profile.empty:
        raise SystemExit("No wells remain after the metadata guards.")

    plates = profile["Metadata_Plate"].to_numpy()
    fit_only = profile.pop("_fit_only").to_numpy(dtype=bool)
    scored = ~fit_only
    report.log(f"\n{int(scored.sum()):,} well(s) to score across "
               f"{profile.loc[scored, 'Metadata_Plate'].nunique():,} plate(s), "
               f"{profile.loc[scored, 'Metadata_Batch'].nunique():,} batch(es); "
               f"{int((control_mask & scored).sum()):,} control well(s), "
               f"{profile.loc[scored, 'Metadata_JCP2022'].nunique():,} distinct "
               f"perturbation(s).")
    if fit_only.any():
        report.log(f"Normalization fitted on {int(control_mask.sum()):,} control "
                   f"well(s) across {profile.loc[control_mask, 'Metadata_Plate'].nunique():,} "
                   f"plate(s), {int(fit_only.sum()):,} of them on plates that "
                   f"are not scored.")

    n_treat = int((~control_mask).sum())
    n_ctrl = int((control_mask & scored).sum())
    rep = profile.loc[~control_mask].groupby("Metadata_JCP2022", observed=True).size()
    whole = int(sum(comb(int(n), 2) for n in rep)) + n_treat * n_ctrl
    report.log(f"A whole-screen well-level activity design would need "
               f"{whole:,} pair(s) (~{whole * BYTES_PER_PAIR / GiB:,.0f} GiB "
               f"peak in copairs).")

    # TVN's CORAL step estimates a covariance per batch from that batch's
    # controls, so thin batches limit the usable rank (about 10 controls per
    # dimension).
    if "Metadata_Source" in profile.columns:
        by_src = (profile.loc[control_mask]
                  .groupby(["Metadata_Source", "Metadata_Batch"], observed=True)
                  .size().groupby("Metadata_Source").agg(["min", "median", "max",
                                                          "size"]))
        by_src.columns = ["controls_min", "controls_median", "controls_max",
                          "n_batches"]
        by_src["max_safe_rank"] = (by_src["controls_min"] // 10).astype(int)
        thin = []
        if str(args.n_components) != "auto":
            rank_now = int(args.n_components)
            column = f"ok_at_rank_{rank_now}"
            by_src[column] = by_src["controls_min"] >= 10 * rank_now
            thin = by_src.index[~by_src[column]].tolist()
        report.log("\nControl wells per batch, by source:")
        report.log(by_src.to_string())
        if thin:
            report.log(f"  WARNING: {', '.join(map(str, thin))} have batches too "
                       f"thin for rank {rank_now} at 10 controls per dimension.")
        report.save_table(by_src.reset_index(), "controls_per_batch_by_source")

    per_batch = (pd.Series(profile.loc[control_mask, "Metadata_Batch"])
                 .value_counts())
    report.log(f"\nControls per batch: min {per_batch.min()}, median "
               f"{per_batch.median():.0f}, max {per_batch.max()} across "
               f"{len(per_batch)} batch(es).")
    report.log(f"Normalizing each block with {args.normalizer.upper()}.")
    args.batch_labels = profile["Metadata_Batch"].to_numpy()
    normalized, usable_columns, degenerate, block_report = normalize_blocks(
        profile, blocks, control_mask, plates, args, report)
    if not normalized:
        raise SystemExit("No usable feature blocks.")
    sets = build_feature_sets(profile, normalized, usable_columns, control_mask,
                              plates, args, report)
    report.log(f"Feature sets to score: {sorted(sets)}")

    n_fit_controls = int(control_mask.sum())
    if fit_only.any():
        profile = profile[scored].reset_index(drop=True)
        sets = {name: values[scored] for name, values in sets.items()}
        report.log(f"Dropped the {int(fit_only.sum()):,} fit-only control "
                   f"well(s) after normalization.")

    info = {
        "excluded_sources": excluded,
        "plate_types": keep_types,
        "fit_controls": args.fit_controls,
        "n_wells": int(profile.shape[0]),
        "n_plates": int(profile["Metadata_Plate"].nunique()),
        "n_batches": int(profile["Metadata_Batch"].nunique()),
        "n_controls": n_ctrl,
        "n_controls_in_fit": n_fit_controls,
        "blocks": block_report,
        "degenerate_blocks": degenerate,
        "whole_screen_pairs": whole,
    }
    return profile, sets, info


# Design B: phenotypic activity (Kalinin et al.)

def phenotypic_activity(meta: pd.DataFrame, values: np.ndarray, label: str,
                        args, cache_dir):
    """Replicate wells retrieved against every DMSO well in `meta`.
    Returns (table, fraction retrieved)."""
    columns = [f"{label}__c{i}" for i in range(values.shape[1])]
    frame = pd.concat(
        [meta[WELL_KEYS + ["Metadata_JCP2022"]].reset_index(drop=True),
         pd.DataFrame(values, columns=columns)], axis=1)
    frame = assign_reference_index(
        frame, f"Metadata_JCP2022 == '{DMSO_JCP2022}'",
        reference_col=REFERENCE_COL, default_value=-1)

    pos_sameby = ["Metadata_JCP2022", REFERENCE_COL]
    average_precision = copairs_map.average_precision(
        frame.filter(regex="^Metadata"), frame[columns].values,
        pos_sameby, [], [], pos_sameby, progress_bar=False)
    average_precision = average_precision.query(
        f"Metadata_JCP2022 != '{DMSO_JCP2022}'")
    table = copairs_map.mean_average_precision(
        average_precision, pos_sameby, null_size=args.null_size,
        threshold=0.05, seed=args.seed, progress_bar=False,
        cache_dir=cache_dir / "copairs_null")
    return table, float(table["below_corrected_p"].mean())


def strict_cohorts(profile, control_mask, n_rep, rng, report=None):
    """
    Kalinin's cpg0004 layout: per compound, up to n_rep replicate wells on
    n_rep distinct plates, with the controls of exactly those plates.
    Compounds on fewer than two plates with controls are skipped.

    Returns a list of (compound, replicate row indices, control row indices).
    """
    plates = profile["Metadata_Plate"].to_numpy()
    compounds = profile["Metadata_JCP2022"].to_numpy()

    controls_on = {}
    for plate in np.unique(plates[control_mask]):
        controls_on[plate] = np.flatnonzero(control_mask & (plates == plate))

    cohorts, short = [], 0
    for compound in np.unique(compounds[~control_mask]):
        rows = np.flatnonzero((compounds == compound) & ~control_mask)
        by_plate = {}
        for row in rows:
            by_plate.setdefault(plates[row], []).append(row)
        usable = [p for p in by_plate if p in controls_on]
        if len(usable) < 2:
            short += 1
            continue
        picked = rng.choice(usable, size=min(n_rep, len(usable)), replace=False)
        reps = np.array([rng.choice(by_plate[p]) for p in picked])
        ctrls = np.concatenate([controls_on[p] for p in picked])
        cohorts.append((compound, reps, ctrls))
    if short and report is not None:
        report.log(f"    {short} compound(s) had <2 plates with controls and "
                   f"were skipped")
    return cohorts


def phenotypic_activity_strict(profile, values, cohorts, label, args, cache_dir):
    """
    Score all strict cohorts in one copairs call.

    Rows are stacked per cohort (a control shared by two compounds appears in
    both) and neg_sameby=["cohort"] confines retrieval to a cohort. One call
    means one mean_average_precision table, so the Benjamini-Hochberg
    correction runs across the whole compound panel, as in Kalinin et al.
    """
    rows, meta_rows = [], []
    for compound, reps, ctrls in cohorts:
        for row in reps:
            rows.append(row); meta_rows.append((compound, compound))
        for row in ctrls:
            rows.append(row); meta_rows.append((compound, DMSO_JCP2022))
    if not rows:
        return None

    rows = np.asarray(rows)
    meta = pd.DataFrame(meta_rows, columns=["cohort", "Metadata_JCP2022"])
    columns = [f"{label}__c{i}" for i in range(values.shape[1])]
    frame = pd.concat([meta, pd.DataFrame(values[rows], columns=columns)], axis=1)
    frame = assign_reference_index(
        frame, f"Metadata_JCP2022 == '{DMSO_JCP2022}'",
        reference_col=REFERENCE_COL, default_value=-1)

    pos_sameby = ["cohort", "Metadata_JCP2022", REFERENCE_COL]
    average_precision = copairs_map.average_precision(
        frame.filter(regex="^(Metadata|cohort)"), frame[columns].values,
        pos_sameby=pos_sameby, pos_diffby=[],
        neg_sameby=["cohort"], neg_diffby=["Metadata_JCP2022", REFERENCE_COL],
        progress_bar=False)
    average_precision = average_precision.query(
        f"Metadata_JCP2022 != '{DMSO_JCP2022}'")
    table = copairs_map.mean_average_precision(
        average_precision, pos_sameby, null_size=args.null_size,
        threshold=0.05, seed=args.seed, progress_bar=False,
        cache_dir=cache_dir / "copairs_null")
    return table, float(table["below_corrected_p"].mean())


def run_kalinin_design(profile, feature_sets, args, report, cache_dir):
    """
    Design B over args.n_draws random draws. Returns (draws, summary, activity).

    strict:     see strict_cohorts.
    subsample:  each draw takes --plates-per-draw whole plates and ranks every
                compound against all their controls, including controls on
                plates that carry none of its replicates.
    """
    report.log("\n=== Design B: phenotypic activity (Kalinin et al. 2025) ===")

    control_mask = (profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()

    if args.kalinin_mode == "strict":
        report.log(f"Strict mode: per compound, {args.replicates_per_compound} "
                   f"replicate well(s) on as many distinct plates, ranked "
                   f"against the DMSO wells of those plates; "
                   f"{args.n_draws} draw(s).")
    else:
        all_plates = np.array(sorted(profile["Metadata_Plate"].unique()))
        n_draw = min(args.plates_per_draw, len(all_plates))
        report.log(f"Subsample mode: {args.n_draws} draw(s) of {n_draw} plate(s) "
                   f"from {len(all_plates)}, pooling every control on the "
                   f"drawn plates.")

    rng = np.random.default_rng(args.seed)
    rows = []
    calls = []
    for draw in range(args.n_draws):
        if args.kalinin_mode == "strict":
            cohorts = strict_cohorts(profile, control_mask,
                                     args.replicates_per_compound, rng,
                                     report if draw == 0 else None)
            if not cohorts:
                report.log("  no compound had >=2 plates carrying controls.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            if draw == 0:
                n_rep = np.mean([len(r) for _, r, _ in cohorts])
                n_ctl = np.mean([len(c) for _, _, c in cohorts])
                report.log(f"  draw 0: {len(cohorts)} compound(s), "
                           f"{n_rep:.1f} replicate(s) and {n_ctl:.0f} "
                           f"plate-matched control(s) each")
            for set_name in sorted(feature_sets):
                values = feature_sets[set_name]
                if is_degenerate(values) is not None:
                    continue
                result = phenotypic_activity_strict(
                    profile, values, cohorts, set_name, args, cache_dir)
                if result is None:
                    continue
                table, fraction = result
                calls.append(pd.DataFrame({
                    "draw": draw, "feature_set": set_name,
                    "Metadata_JCP2022": table["Metadata_JCP2022"].to_numpy(),
                    "retrieved": table["below_corrected_p"].to_numpy().astype(bool),
                }))
                rows.append({
                    "draw": draw, "feature_set": set_name,
                    "n_plates": args.replicates_per_compound,
                    "n_treatment_wells": int(sum(len(r) for _, r, _ in cohorts)),
                    "n_control_wells": int(sum(len(c) for _, _, c in cohorts)),
                    "median_replicates": float(np.median(
                        [len(r) for _, r, _ in cohorts])),
                    "n_compounds_scored": int(len(table)),
                    "mean_map": float(table["mean_average_precision"].mean()),
                    "median_map": float(table["mean_average_precision"].median()),
                    "percent_retrieved": 100 * fraction,
                })
            if (draw + 1) % max(1, args.n_draws // 10) == 0:
                report.log(f"  ... {draw + 1}/{args.n_draws} draw(s) done")
            continue

        picked = set(rng.choice(all_plates, size=n_draw, replace=False).tolist())
        mask = profile["Metadata_Plate"].isin(picked).to_numpy()
        sub_meta = profile.loc[mask].reset_index(drop=True)

        is_control = (sub_meta["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()
        n_ctrl = int(is_control.sum())
        replicate_counts = (sub_meta.loc[~is_control]
                            .groupby("Metadata_JCP2022", observed=True).size())
        scorable = replicate_counts[replicate_counts >= 2]
        if n_ctrl == 0 or scorable.empty:
            report.log(f"  draw {draw}: no controls or no compound with >=2 "
                       f"replicates; skipped.")
            continue

        n_treat = int((~is_control).sum())
        n_pos = int(sum(comb(int(n), 2) for n in replicate_counts))
        n_neg = n_treat * n_ctrl
        if draw == 0:
            report.log(f"  draw 0 shape: {n_treat:,} treatment + {n_ctrl:,} control "
                       f"well(s); median {replicate_counts.median():.0f} replicate(s) "
                       f"per compound; {len(scorable)} scorable compound(s)")
            if not report_budget(n_pos + n_neg, "per draw",
                                 args.memory_budget_gib, report):
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for set_name in sorted(feature_sets):
            values = feature_sets[set_name][mask]
            if is_degenerate(values) is not None:
                continue
            table, fraction = phenotypic_activity(
                sub_meta, values, set_name, args, cache_dir)
            calls.append(pd.DataFrame({
                "draw": draw,
                "feature_set": set_name,
                "Metadata_JCP2022": table["Metadata_JCP2022"].to_numpy(),
                "retrieved": table["below_corrected_p"].to_numpy().astype(bool),
            }))
            rows.append({
                "draw": draw,
                "feature_set": set_name,
                "n_plates": n_draw,
                "n_treatment_wells": n_treat,
                "n_control_wells": n_ctrl,
                "median_replicates": float(replicate_counts.median()),
                "n_compounds_scored": int(len(table)),
                "mean_map": float(table["mean_average_precision"].mean()),
                "median_map": float(table["mean_average_precision"].median()),
                "percent_retrieved": 100 * fraction,
            })
        if (draw + 1) % max(1, args.n_draws // 10) == 0:
            report.log(f"  ... {draw + 1}/{args.n_draws} draw(s) done")

    draws = pd.DataFrame(rows)
    if draws.empty:
        report.log("No draw produced a scorable design.")
        return draws, pd.DataFrame(), pd.DataFrame()

    report.save_table(draws, "kalinin_draws")
    summary = (draws.groupby("feature_set")
               .agg(n_draws=("percent_retrieved", "size"),
                    retrieved_mean=("percent_retrieved", "mean"),
                    retrieved_sd=("percent_retrieved", "std"),
                    retrieved_p05=("percent_retrieved", lambda s: s.quantile(0.05)),
                    retrieved_p95=("percent_retrieved", lambda s: s.quantile(0.95)),
                    map_mean=("mean_map", "mean"),
                    map_sd=("mean_map", "std"))
               .reset_index()
               .sort_values("retrieved_mean", ascending=False)
               .reset_index(drop=True))
    report.save_table(summary, "kalinin_summary")
    report.log("\n" + summary.to_string(index=False))

    activity = summarize_activity(pd.concat(calls, ignore_index=True), args, report)
    return draws, summary, activity


def summarize_activity(calls: pd.DataFrame, args, report) -> pd.DataFrame:
    """
    One activity call per (feature set, compound): active when retrieved in at
    least --activity-min-rate of the draws in which the compound was scorable.
    """
    grouped = (calls.groupby(["feature_set", "Metadata_JCP2022"], observed=True)
               .agg(n_draws_scored=("retrieved", "size"),
                    n_draws_retrieved=("retrieved", "sum"))
               .reset_index())
    grouped["retrieval_rate"] = (grouped["n_draws_retrieved"]
                                 / grouped["n_draws_scored"].clip(lower=1))
    grouped["is_active"] = grouped["retrieval_rate"] >= args.activity_min_rate

    report.log(f"\nPhenotypic activity calls (retrieved in >= "
               f"{100 * args.activity_min_rate:.0f}% of the draws in which the "
               f"compound was scorable):")
    for set_name, part in grouped.groupby("feature_set", observed=True):
        n_active = int(part["is_active"].sum())
        report.log(f"  {set_name:<20} {n_active:>4} / {len(part):>4} compound(s) "
                   f"active ({100 * n_active / max(len(part), 1):.1f}%)")

    report.save_table(grouped, "kalinin_activity")
    return grouped


def select_comparable_sources(profile, n_rep, coverage, seed, log):
    """
    Sources that can score at least `coverage` of the compound panel (a
    compound is scorable in a source when it has >=2 plates with controls
    there), the panel common to all of them, and a diagnostics table for every
    source.

    Returns (usable, common, diag); `usable` maps source -> boolean row mask.
    """
    sources = profile["Metadata_Source"].to_numpy()
    plates = profile["Metadata_Plate"].to_numpy()
    compounds = profile["Metadata_JCP2022"].to_numpy()
    panel = set(np.unique(compounds[compounds != DMSO_JCP2022]))

    rows, qualifying = [], {}
    for source in sorted(pd.unique(sources)):
        in_source = sources == source
        sub_profile = profile.loc[in_source].reset_index(drop=True)
        sub_control = (sub_profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()
        present = set(np.unique(compounds[in_source])) & panel

        scorable = set()
        if sub_control.any():
            scorable = {c for c, _, _ in strict_cohorts(
                sub_profile, sub_control, n_rep, np.random.default_rng(seed))}
        frac = len(scorable) / max(len(panel), 1)
        rows.append({
            "Metadata_Source": source,
            "n_plates": int(len(np.unique(plates[in_source]))),
            "n_wells": int(in_source.sum()),
            "n_control_wells": int(sub_control.sum()),
            "n_compounds_present": len(present),
            "n_compounds_scorable": len(scorable),
            "coverage": frac,
        })
        if frac >= coverage:
            qualifying[source] = (in_source, scorable)

    diag = pd.DataFrame(rows).sort_values("coverage", ascending=False)
    log("\n" + diag.to_string(index=False))

    if not qualifying:
        best = diag["coverage"].max() if not diag.empty else 0.0
        log(f"\nWARNING: no source can score {100 * coverage:.0f}% of the "
            f"{len(panel)}-compound panel; the best is {100 * best:.1f}%. "
            f"Skipping the per-source analysis. Lower --source-coverage to "
            f"include partial sources, which are then restricted to their "
            f"common panel.")
        return {}, set(), diag

    common = set.intersection(*(sc for _, sc in qualifying.values()))
    if common != panel:
        log(f"\nRestricting every source to the {len(common)} compound(s) "
            f"scorable in all {len(qualifying)} qualifying source(s) "
            f"({len(panel) - len(common)} dropped).")
    log(f"\nUsing {len(qualifying)} source(s): {', '.join(sorted(qualifying))}")
    return {k: v[0] for k, v in qualifying.items()}, common, diag


def run_source_stratified(profile, feature_sets, args, report, cache_dir):
    """
    Strict Design B within each imaging source, on the common compound panel.
    The normalization is the one fitted on the whole run; only the scoring is
    restricted. Replicate counts may still differ between sources.
    """
    if args.stratify_source != "on":
        return pd.DataFrame()
    if "Metadata_Source" not in profile.columns:
        report.log("\nNo Metadata_Source column; skipping source stratification.")
        return pd.DataFrame()

    report.log("\n=== Design B, per imaging source ===")
    usable, common, diag = select_comparable_sources(
        profile, args.replicates_per_compound, args.source_coverage,
        args.seed, report.log)
    report.save_table(diag, "kalinin_source_diagnostics")
    if not usable:
        return pd.DataFrame()

    rows = []
    for source, in_source in usable.items():
        sub_profile = profile.loc[in_source].reset_index(drop=True)
        sub_control = (sub_profile["Metadata_JCP2022"] == DMSO_JCP2022).to_numpy()
        rng = np.random.default_rng(args.seed)
        for draw in range(args.n_draws):
            cohorts = [c for c in strict_cohorts(sub_profile, sub_control,
                                                 args.replicates_per_compound, rng)
                       if c[0] in common]
            if not cohorts:
                break
            median_reps = float(np.median([len(r) for _, r, _ in cohorts]))
            for set_name in sorted(feature_sets):
                values = feature_sets[set_name][in_source]
                if is_degenerate(values) is not None:
                    continue
                result = phenotypic_activity_strict(
                    sub_profile, values, cohorts, set_name, args, cache_dir)
                if result is None:
                    continue
                table, fraction = result
                rows.append({
                    "Metadata_Source": source, "draw": draw,
                    "feature_set": set_name,
                    "n_compounds_scored": int(len(table)),
                    "median_replicates": median_reps,
                    "mean_map": float(table["mean_average_precision"].mean()),
                    "percent_retrieved": 100 * fraction,
                })

    draws = pd.DataFrame(rows)
    if draws.empty:
        report.log("  nothing scored.")
        return draws

    report.save_table(draws, "kalinin_by_source_draws")
    summary = (draws.groupby(["Metadata_Source", "feature_set"])
               .agg(n_draws=("percent_retrieved", "size"),
                    n_compounds=("n_compounds_scored", "median"),
                    median_replicates=("median_replicates", "median"),
                    retrieved_mean=("percent_retrieved", "mean"),
                    retrieved_sd=("percent_retrieved", "std"),
                    map_mean=("mean_map", "mean"))
               .reset_index()
               .sort_values(["feature_set", "retrieved_mean"], ascending=[True, False])
               .reset_index(drop=True))
    report.save_table(summary, "kalinin_by_source")
    report.log("\n" + summary.to_string(index=False))
    return summary


# Figures. pyplot is imported here, after the caller has chosen a backend.

def figure_kalinin(draws, report):
    if draws.empty:
        return
    import matplotlib.pyplot as plt
    order = (draws.groupby("feature_set")["percent_retrieved"].mean()
             .sort_values(ascending=False).index.tolist())
    data = [draws.loc[draws["feature_set"] == name, "percent_retrieved"].to_numpy()
            for name in order]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(order)), 4.5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(values))
        ax.plot(i + jitter, values, ".", alpha=0.4, markersize=4, color="tab:blue")
    ax.set_ylabel("% compounds retrieved")
    n_plates = int(draws["n_plates"].iloc[0])
    ax.set_title(f"Phenotypic activity over {draws['draw'].nunique()} draws "
                 f"of {n_plates} plates")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    report.finish_figure(fig, "kalinin_draws")
    plt.close(fig)
