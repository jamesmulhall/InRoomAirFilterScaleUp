"""Load and transform raw inputs for the essential-worker pipeline."""

from __future__ import annotations

import warnings
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


def location_to_indoor_fraction(location: float, *, partial: bool = True) -> float:
    """
    Map JEM Location to an indoor fraction using the nearest bucket.

    Arguments:
        location (float): JEM Location score (0–3).
        partial (bool): If True, bucket 2 → 50%; if False, buckets 2 and 3 → 100%.

    Returns:
        float: Indoor fraction in [0, 1], or NaN when location is missing.
    """
    if partial:
        buckets = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}
    else:
        buckets = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
    if pd.isna(location):
        return float("nan")
    nearest = min(buckets, key=lambda k: abs(float(location) - k))
    return buckets[nearest]


def load_jem_l4_indoor_fraction(
    jem_path: Path,
    *,
    partial: bool = True,
) -> Dict[str, float]:
    """
    Mean Location across JEM country sheets → indoor fraction per ISCO L4.

    Arguments:
        jem_path (Path): Path to ``job_exposure_matrix.xls``.
        partial (bool): Passed to :func:`location_to_indoor_fraction`.

    Returns:
        dict: ISCO L4 code → indoor fraction.
    """
    sheets = pd.read_excel(jem_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df["ISCO-08"] = df["ISCO-08"].astype(str).str.zfill(4)
    loc = df.groupby("ISCO-08")["Location"].mean()
    return {
        str(k): location_to_indoor_fraction(v, partial=partial) for k, v in loc.items()
    }


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

    if indoor_context_method in ("jem_partial", "jem_binary"):
        if jem_path is None:
            raise ValueError(
                "jem_path is required for indoor_context_method="
                f"{indoor_context_method!r}"
            )
        l4_indoor = load_jem_l4_indoor_fraction(
            jem_path, partial=(indoor_context_method == "jem_partial")
        )
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


def _parse_isco_l2_code(classif_label: str) -> str:
    """Normalised ISCO-08 L2 key from ``classif1.label``: ``22``, ``Tot`` or ``Not``."""
    part = str(classif_label).split(":", 1)[-1].strip()
    if part.lower().startswith("total"):
        return "Tot"
    if part.lower().startswith("not"):
        return "Not"
    return part[:2] if part[:2].isdigit() else part[:3].strip()


def _nec_pct_for_year(year_codes: Dict[str, float]) -> Optional[float]:
    tot = year_codes.get("Tot")
    nec = year_codes.get("Not")
    if tot is None or not pd.notna(tot) or tot <= 0:
        return None
    if nec is None or not pd.notna(nec):
        return 0.0
    return float(nec) / float(tot)


def _select_country_ilo_year(
    years: Dict[int, Dict[str, float]],
) -> Optional[int]:
    """Pick one survey year: latest with NEC/Tot <= 10%, else minimum NEC %."""
    if not years:
        return None
    sorted_years = sorted(years)
    if not any(_nec_pct_for_year(years[y]) is not None for y in sorted_years):
        return sorted_years[-1]
    for year in reversed(sorted_years):
        nec_pct = _nec_pct_for_year(years[year])
        if nec_pct is not None and nec_pct <= 0.10:
            return year
    best_year = sorted_years[0]
    best_nec = float("inf")
    for year in sorted_years:
        nec_pct = _nec_pct_for_year(years[year])
        if nec_pct is None:
            continue
        if nec_pct < best_nec or (nec_pct == best_nec and year > best_year):
            best_nec = nec_pct
            best_year = year
    return best_year


ARMED_FORCES_ISCO_CODES = ("01", "02", "03")
SECURITY_ISCO_CODE = "54"
CLEANING_ISCO_CODES = ("91", "96")
LAOS_COUNTRY = "Laos"
US_COUNTRY = "United States"

# Active-duty end strength from the DOD 2023 Demographics Profile (2024).
# Officers (Table 2.01) map to ISCO 01. Enlisted (1,038,909) is split by pay
# grade (Congressional Research Service IF10684, March 2024), scaled to the DOD
# enlisted total: E-5 through E-9 → ISCO 02, E-1 through E-4 → ISCO 03.
# Total active duty = 1,273,382.
US_ARMED_FORCES_EMPLOYMENT = {
    "01": 234_473,
    "02": 519_571,
    "03": 519_338,
}


def _coded_employment(emp: Dict[str, float]) -> float:
    """Sum reported ISCO L2 employment, excluding total and NEC rows."""
    return sum(
        float(v)
        for k, v in emp.items()
        if k not in ("Tot", "Not") and pd.notna(v) and float(v) > 0
    )


def _isco_code_missing(emp: Dict[str, float], code: str) -> bool:
    """True when an ISCO code is absent or has no usable employment."""
    value = emp.get(code)
    return value is None or not pd.notna(value) or float(value) <= 0


def _median_isco_shares(
    nested: Dict[str, Dict[str, float]], codes: tuple[str, ...]
) -> Dict[str, float]:
    """
    Median share of coded employment for each ISCO code across reporting countries.

    Arguments:
        nested (dict): Country employment from :func:`build_employment_by_isco`.
        codes (tuple): ISCO L2 codes to summarise.

    Returns:
        dict: Median employment share for each code.
    """
    shares: Dict[str, list[float]] = {code: [] for code in codes}
    for emp in nested.values():
        coded = _coded_employment(emp)
        if coded <= 0:
            continue
        for code in codes:
            value = emp.get(code)
            if value is not None and pd.notna(value) and float(value) > 0:
                shares[code].append(float(value) / coded)
    return {
        code: float(np.median(values)) if values else 0.0
        for code, values in shares.items()
    }


def _apply_us_armed_forces_override(
    nested: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Replace US armed-forces ISCO headcounts with DOD active-duty end strength.

    ILO does not report ISCO 01–03 for the United States. Median-share imputation
    understates active duty, so we use DOD demographics instead.

    Arguments:
        nested (dict): Country employment after other imputation steps.

    Returns:
        dict: The same structure with US codes 01–03 set from DoD sources.
    """
    emp = nested.get(US_COUNTRY)
    if emp is None:
        return nested
    for code in ARMED_FORCES_ISCO_CODES:
        emp[code] = float(US_ARMED_FORCES_EMPLOYMENT[code])
    return nested


def impute_missing_ilo_employment(
    nested: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Fill missing ILO ISCO headcounts using global median shares of coded employment.

    Armed forces (01–03) and security (54) are imputed for every country with a
    gap except the United States, which uses DOD active-duty end strength.
    Cleaning (91, 96) is imputed for Laos only.

    Arguments:
        nested (dict): Country employment from the ILO ISCO table.

    Returns:
        dict: The same structure with imputed codes added in place.
    """
    armed_shares = _median_isco_shares(nested, ARMED_FORCES_ISCO_CODES)
    security_share = _median_isco_shares(nested, (SECURITY_ISCO_CODE,))[
        SECURITY_ISCO_CODE
    ]
    cleaning_shares = _median_isco_shares(nested, CLEANING_ISCO_CODES)

    for country, emp in nested.items():
        coded = _coded_employment(emp)
        if coded <= 0:
            continue
        if country != US_COUNTRY:
            for code in ARMED_FORCES_ISCO_CODES:
                if _isco_code_missing(emp, code) and armed_shares[code] > 0:
                    emp[code] = coded * armed_shares[code]
        if _isco_code_missing(emp, SECURITY_ISCO_CODE) and security_share > 0:
            emp[SECURITY_ISCO_CODE] = coded * security_share
        if country == LAOS_COUNTRY:
            for code in CLEANING_ISCO_CODES:
                if _isco_code_missing(emp, code) and cleaning_shares[code] > 0:
                    emp[code] = coded * cleaning_shares[code]
    return _apply_us_armed_forces_override(nested)


def build_employment_by_isco(ilo_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Reduce ILO ISCO-08 L2 employment CSV to ``{country: {code: employment}}``.

    - Keeps only ``sex.label == "Total"`` when that column exists.
    - Uses one survey year per country (all codes from the same year).
    - Year: latest with NEC share of total <= 10%; if none qualify, the year
      with the lowest NEC share (ties → more recent year).
    - Missing armed forces (01–03) and security (54) are imputed from global
      median shares of coded employment; the United States uses DOD active-duty
      end strength for 01–03. Cleaning (91, 96) is imputed for Laos only.
    """
    df = ilo_df.copy()
    if "sex.label" in df.columns:
        df = df[df["sex.label"] == "Total"].copy()
    else:
        warnings.warn(
            "ILO employment CSV has no sex.label column; sex breakdown not filtered.",
            stacklevel=2,
        )

    df["ISCO-8 L2 Code"] = df["classif1.label"].map(_parse_isco_l2_code)
    df["Employment"] = pd.to_numeric(df["obs_value"], errors="coerce") * 1000
    df = df.rename(columns={"ref_area.label": "Country"})[
        ["Country", "ISCO-8 L2 Code", "Employment", "time"]
    ]
    df = df[df["Employment"].notna()]

    by_country_year: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        year = int(row["time"])
        by_country_year.setdefault(row["Country"], {}).setdefault(year, {})[
            row["ISCO-8 L2 Code"]
        ] = float(row["Employment"])

    by_country_year = normalise_country_keys(by_country_year)

    nested: Dict[str, Dict[str, float]] = {}
    for country, years in by_country_year.items():
        year = _select_country_ilo_year(years)
        if year is None:
            continue
        nested[country] = dict(years[year])

    return impute_missing_ilo_employment(nested)


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
