"""Load and transform raw inputs for the essential-worker pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from essential_workers import IndoorContextMethod

import numpy as np
import pandas as pd

try:
    import country_converter as coco
except ImportError as exc:  # pragma: no cover - required runtime dep
    raise ImportError(
        "country_converter is required. Install via "
        "`mamba install country_converter`"
    ) from exc

_CC = coco.CountryConverter()


def normalise_country_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Standardise the top-level country keys of a nested dict."""
    return {
        _CC.convert(names=k, to="name_short", not_found="not found"): v
        for k, v in d.items()
    }


def employment_for_country(
    country: str,
    employment_by_iso: Dict[str, Dict[str, float]],
    employment_country_aliases: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, float]]:
    """Look up ILO ISCO employment for a labour-force country name."""
    if country in employment_by_iso:
        return employment_by_iso[country]
    aliases = employment_country_aliases or {}
    alias = aliases.get(country)
    if alias and alias in employment_by_iso:
        return employment_by_iso[alias]
    return None


def normalize_soc(code: str) -> str:
    """Strip O*NET suffix (e.g. ``51-4071.00`` → ``51-4071``)."""
    return str(code).split(".")[0]


def merge_onet_max_context(
    onet_controlled_df: pd.DataFrame,
    onet_not_controlled_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-SOC indoor % (0–100) = max of environmentally controlled vs not."""
    env = onet_controlled_df[["Code", "Context"]].copy()
    not_env = onet_not_controlled_df[["Code", "Context"]].copy()
    env["Code"] = env["Code"].map(normalize_soc)
    not_env["Code"] = not_env["Code"].map(normalize_soc)
    merged = env.merge(not_env, on="Code", how="outer", suffixes=("_env", "_not"))
    merged["context_pct"] = merged[["Context_env", "Context_not"]].max(axis=1)
    return merged[["Code", "context_pct"]]


def pct_to_indoor_fraction(
    context_pct: float,
    method: IndoorContextMethod,
) -> float:
    """Map O*NET indoor % to a 0..1 fraction."""
    if method == "onet_max":
        return float(context_pct) / 100.0
    if context_pct >= 75:
        return 1.0
    if context_pct >= 50:
        return 0.5
    return 0.0


def location_to_indoor_fraction(location: float) -> float:
    """JEM Location scale: nearest of 0/1 → 0%, 2 → 50%, 3 → 100%."""
    buckets = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}
    if pd.isna(location):
        return float("nan")
    nearest = min(buckets, key=lambda k: abs(float(location) - k))
    return buckets[nearest]


def load_jem_l4_indoor_fraction(jem_path: Path) -> Dict[str, float]:
    """Mean Location across JEM country sheets → indoor fraction per ISCO L4."""
    sheets = pd.read_excel(jem_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df["ISCO-08"] = df["ISCO-08"].astype(str).str.zfill(4)
    loc = df.groupby("ISCO-08")["Location"].mean()
    return {str(k): location_to_indoor_fraction(v) for k, v in loc.items()}


def onet_l4_indoor_fraction(
    onet_controlled_df: pd.DataFrame,
    onet_not_controlled_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    soc_to_isco_aggregator: str,
    indoor_context_method: IndoorContextMethod,
) -> Dict[str, float]:
    """SOC max-O*NET % → ISCO L4 indoor fraction via crosswalk."""
    soc_df = merge_onet_max_context(onet_controlled_df, onet_not_controlled_df)
    cw = crosswalk_df[["2010 SOC Code", "ISCO-08 Code"]].copy()
    cw["ISCO-08 Code"] = cw["ISCO-08 Code"].astype(str)
    soc_to_isco = cw.set_index("2010 SOC Code")["ISCO-08 Code"].to_dict()

    pool: Dict[str, list] = {}
    for _, row in soc_df.iterrows():
        soc = row["Code"]
        if soc not in soc_to_isco:
            continue
        isco = soc_to_isco[soc]
        frac = pct_to_indoor_fraction(row["context_pct"], indoor_context_method)
        pool.setdefault(isco, []).append(frac)

    if soc_to_isco_aggregator == "mean":
        return {k: float(np.mean(v)) for k, v in pool.items()}
    return {k: v[-1] for k, v in pool.items()}


def poll_l4_with_indoors_context(
    poll_df: pd.DataFrame,
    l4_indoor: Dict[str, float],
) -> pd.DataFrame:
    """Attach ``indoors_context`` at ISCO L4; fill hierarchy; ready for L2 collapse."""
    from essential_workers import INDOORS_CONTEXT_COLUMN, OVERRIDES_INDOOR_L4

    poll = poll_df[["ISCO-08", "Census"]].copy()
    poll["ISCO-08"] = poll["ISCO-08"].astype(str).str.zfill(4)
    poll = poll.set_index("ISCO-08")
    poll["Census"] = pd.to_numeric(poll["Census"], errors="coerce")

    df = poll.copy()
    df[INDOORS_CONTEXT_COLUMN] = df.index.map(l4_indoor)

    df["_L3"] = df.index.str[:3]
    df["_L2"] = df.index.str[:2]
    df["_L1"] = df.index.str[:1]
    mean_l3 = df.groupby("_L3")[INDOORS_CONTEXT_COLUMN].transform("mean")
    mean_l2 = df.groupby("_L2")[INDOORS_CONTEXT_COLUMN].transform("mean")
    mean_l1 = df.groupby("_L1")[INDOORS_CONTEXT_COLUMN].transform("mean")
    df[INDOORS_CONTEXT_COLUMN] = (
        df[INDOORS_CONTEXT_COLUMN].fillna(mean_l3).fillna(mean_l2).fillna(mean_l1)
    )

    overrides_present = [c for c in OVERRIDES_INDOOR_L4 if c in df.index]
    df.loc[overrides_present, INDOORS_CONTEXT_COLUMN] = 1.0
    df = df.drop(columns=["_L3", "_L2", "_L1"])
    return df


def collapse_poll_l4_to_lvl2(l4_df: pd.DataFrame) -> pd.DataFrame:
    """Average poll vital weight and indoors_context to ISCO L2."""
    from essential_workers import (
        ILO_LVL2_ESSENTIAL_GROUPS,
        INDOORS_CONTEXT_COLUMN,
        ISCO_L2_TO_GROUP,
        NON_ILO_POLL_CODES,
    )

    lvl2 = (
        l4_df.assign(_l2=l4_df.index.str[:2])
        .groupby("_l2")
        .agg({"Census": "mean", INDOORS_CONTEXT_COLUMN: "mean"})
        .rename_axis("ISCO-08")
        .rename(columns={"Census": "Vital Weight POLL"})
    )
    lvl2["Essential Weight ILO"] = (
        lvl2.index.astype(int).isin(ILO_LVL2_ESSENTIAL_GROUPS).astype(int)
    )
    if "62" in lvl2.index:
        lvl2.loc["61"] = lvl2.loc["62"]
    if "63" in lvl2.index:
        lvl2.at["63", INDOORS_CONTEXT_COLUMN] = 0.0
    for code in NON_ILO_POLL_CODES:
        if code in lvl2.index:
            lvl2.at[code, "Vital Weight POLL"] = 0
    lvl2["Group"] = lvl2.index.map(ISCO_L2_TO_GROUP)
    return lvl2


def build_isco_lvl2_template(
    poll_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    onet_controlled_df: Optional[pd.DataFrame] = None,
    onet_not_controlled_df: Optional[pd.DataFrame] = None,
    *,
    indoor_context_method: IndoorContextMethod = "onet_max",
    jem_path: Optional[Path] = None,
    soc_to_isco_aggregator: str = "mean",
) -> pd.DataFrame:
    """ISCO L2 table before group overlaps (poll, indoors_context, essential, group)."""
    if soc_to_isco_aggregator not in ("mean", "last"):
        raise ValueError(
            f"soc_to_isco_aggregator must be 'mean' or 'last', got "
            f"{soc_to_isco_aggregator!r}"
        )

    if indoor_context_method == "jem_location":
        if jem_path is None:
            raise ValueError(
                "jem_path is required for indoor_context_method='jem_location'"
            )
        l4_indoor = load_jem_l4_indoor_fraction(jem_path)
    else:
        if onet_controlled_df is None or onet_not_controlled_df is None:
            raise ValueError(
                "onet_controlled_df and onet_not_controlled_df are required "
                f"for indoor_context_method={indoor_context_method!r}"
            )
        l4_indoor = onet_l4_indoor_fraction(
            onet_controlled_df,
            onet_not_controlled_df,
            crosswalk_df,
            soc_to_isco_aggregator,
            indoor_context_method,
        )

    l4_df = poll_l4_with_indoors_context(poll_df, l4_indoor)
    return collapse_poll_l4_to_lvl2(l4_df)


def build_employment_by_isco(ilo_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Reduce ILO ISCO-08 L2 employment CSV to ``{country: {code: employment}}``."""
    df = ilo_df.copy()
    df["ISCO-8 L2 Code"] = df["classif1.label"].str.split(":").str[1].str[1:4]
    df["Employment"] = df["obs_value"] * 1000
    df = df.rename(columns={"ref_area.label": "Country"})[
        ["Country", "ISCO-8 L2 Code", "Employment", "time"]
    ]

    nested: Dict[str, Dict[str, Dict[Any, float]]] = {}
    for _, row in df.iterrows():
        nested.setdefault(row["Country"], {}).setdefault(row["ISCO-8 L2 Code"], {})[
            row["time"]
        ] = row["Employment"]

    nested = normalise_country_keys(nested)

    for country, code_dict in nested.items():
        for code, year_dict in list(code_dict.items()):
            if isinstance(year_dict, dict) and year_dict:
                valid = {y: v for y, v in year_dict.items() if pd.notna(v)}
                code_dict[code] = valid[max(valid)] if valid else None

    return nested  # type: ignore[return-value]


def load_ilo_published_pct(path: Path) -> pd.DataFrame:
    """Read and standardise the ILO 2023 per-country %essential xlsx."""
    df = pd.read_excel(path, sheet_name="Sheet1", header=1, engine="openpyxl")
    df = df.rename(
        columns={
            "cname": "Country Name",
            "Share of key workers": "ILO %essential (published)",
            "Same share without agriculture": "ILO %essential non-agri (published)",
        }
    )
    df = df[
        [
            "Country Name",
            "ILO %essential (published)",
            "ILO %essential non-agri (published)",
        ]
    ]
    df = df[df["Country Name"].notna() & (df["Country Name"] != "Average")].copy()
    df["Country Name"] = _CC.convert(
        df["Country Name"].tolist(), to="name_short", not_found="not found"
    )
    df["ILO %essential (published)"] = pd.to_numeric(
        df["ILO %essential (published)"], errors="coerce"
    )
    df["ILO %essential non-agri (published)"] = pd.to_numeric(
        df["ILO %essential non-agri (published)"], errors="coerce"
    )
    return df


def prepare_labour_force(lf_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the WB labour-force dataframe and attach UN regions."""
    df = lf_df.copy()
    df["Country Name"] = _CC.convert(
        df["Country Name"], to="short_name", not_found="not found"
    )
    df["Region"] = _CC.convert(df["Country Name"], to="UNregion", not_found="not found")
    return df
