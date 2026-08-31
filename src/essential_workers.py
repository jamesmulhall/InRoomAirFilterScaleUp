"""Essential / Indoor worker estimation pipeline.

This module re-implements the analysis previously embedded in
``scripts/Essential_Worker_Processing.ipynb`` as a small set of pure
functions plus a :func:`run_pipeline` orchestrator. The pipeline turns the
ILO ISCO-08 employment data, the team's ISCO-08 vital/essential poll,
ONET indoors context, the SOC-ISCO crosswalk and the World Bank labour
force figures into per-country and per-region counts of indoor / total
vital / essential workers. It also exposes :func:`validate_against_ilo`
for comparing the per-country result against the ILO 2023 published
"share of key workers" figures.

The notebook is now a thin walkthrough that imports from here.

Methodology overview
--------------------
The reference for "essential worker" classification is:

    ILO (2023). *World Employment and Social Outlook 2023: The value of
    essential work*. International Labour Organization.
    https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@dcomm/@publ/documents/publication/wcms_871016.pdf

The ILO's definition is a **two-axis intersection**:

    A worker is a "key worker" iff they are
        (a) in a key OCCUPATION  (ISCO-08 code in Table A2 of the report)
                                 AND
        (b) in a key INDUSTRY    (ISIC Rev.4 code in Table A1).

The ILO produces its per-country published shares from worker-level
microdata, so each respondent's ISCO ∩ ISIC status is known exactly.

**This repository does not have access to ISCO × ISIC cross-tabulated
microdata.** All we have at country level is the marginal: ILO employment
broken down by ISCO-08 L2 code (``ILO_ISCO_08_GLB.csv``). To approximate
the ISCO ∩ ISIC intersection we therefore make a single, important
simplification:

    For every occupational group g (Food, Health, Retail, Security,
    Transport, Manual, Cleaning, Tech, ArmedForces) we multiply that
    country's per-ISCO employment by a *global* overlap factor
    ``GROUP_OVERLAP[g]`` derived from ILO Figure A1 / Table B2.
    ``GROUP_OVERLAP[g]`` is the global aggregate fraction of workers in
    occupational group g that are also in a key ISIC industry.

**Per-country calibration (default pipeline).** Global
:data:`GROUP_OVERLAP` values are Figure A1 priors. For each country with
ILO ISCO employment and a published WESO %essential, a scalar ``x ∈ [0, 1]``
adjusts all eight calibratable groups together (toward 1 when raising,
toward 0 when lowering); Armed Forces stays at 0.40. Vital and essential
totals both use the calibrated overlaps. Countries without ILO microdata
receive neighbour-averaged overlaps via :data:`SIMILAR_ISO3` (not global
priors). Where a single ``x`` cannot reach the ILO target even at
``x = 1``, the solver flags ``infeasible_clipped`` (documented in
``Group_Overlap_Calibration.csv``).

Before calibration, assuming identical overlap structure across countries
was the dominant source of deviation from ILO published shares; the pipeline
logs both model (global overlap) and calibrated series in
``Essential_Workers_Validation.csv``.

Two other simplifications worth flagging:

1. **Armed forces overlap factor.** ILO Figure A1 only covers the 8
   ISIC-classifiable occupational groups. Armed Forces (ISCO codes
   01/02/03) are reported separately and excluded from ILO's headline
   global figures. We retain them but assign an overlap factor of 0.40
   (Blueprint Biosecurity assumption) so that downstream indoor-essential
   counts include uniformed services. See ``GROUP_OVERLAP['ArmedForces']``.

2. **Teleworkable exclusions.** ILO Table A2 explicitly excludes a
   handful of ISCO L2 codes the report classifies as teleworkable. The
   team's in-house poll flagged some of these as vital anyway; to keep
   our Vital figures aligned with the ILO methodology we zero those
   codes via :data:`NON_ILO_POLL_CODES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from paths import ESSENTIAL_WORKERS_DATA, ESSENTIAL_WORKERS_RESULTS

# ---------------------------------------------------------------------------
# Constants (domain / ISCO weights — used by preprocessing via lazy import)
# ---------------------------------------------------------------------------

IndoorContextMethod = Literal["onet_max", "onet_banded", "jem_location"]
INDOORS_CONTEXT_COLUMN = "indoors_context"

# ISCO-08 level-2 codes the ILO classes as essential (WESO 2023 Table A2).
ILO_LVL2_ESSENTIAL_GROUPS = [
    61,
    62,
    63,
    92,
    94,
    22,
    32,
    53,
    52,
    95,
    54,
    71,
    72,
    73,
    74,
    75,
    81,
    82,
    93,
    91,
    96,
    83,
    31,
    44,
    51,
    1,
    2,
    3,
]

ISCO_L2_TO_GROUP: Dict[str, str] = {
    "61": "Food",
    "62": "Food",
    "63": "Food",
    "92": "Food",
    "94": "Food",
    "22": "Health",
    "32": "Health",
    "53": "Health",
    "52": "Retail",
    "95": "Retail",
    "54": "Security",
    "71": "Manual",
    "72": "Manual",
    "73": "Manual",
    "74": "Manual",
    "75": "Manual",
    "81": "Manual",
    "82": "Manual",
    "93": "Manual",
    "91": "Cleaning",
    "96": "Cleaning",
    "83": "Transport",
    "31": "Tech",
    "44": "Tech",
    "51": "Tech",
    "01": "ArmedForces",
    "02": "ArmedForces",
    "03": "ArmedForces",
}

NON_ILO_POLL_CODES = ["13", "21", "33", "35"]
OVERRIDES_INDOOR_L4 = ["0110", "0210", "0310"]

from preprocessing import (  # noqa: E402,F401
    build_employment_by_isco,
    build_isco_lvl2_template,
    build_isco_lvl2_template as _build_isco_lvl2_template,
    employment_for_country,
    load_ilo_published_pct,
    location_to_indoor_fraction as _location_to_indoor_fraction,
    merge_onet_max_context as _merge_onet_max_context,
    pct_to_indoor_fraction as _pct_to_indoor_fraction,
    prepare_labour_force,
)

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

UN_REGIONS = [
    "Australia and New Zealand",
    "Caribbean",
    "Central America",
    "Central Asia",
    "Eastern Africa",
    "Eastern Asia",
    "Eastern Europe",
    "Melanesia",
    "Micronesia",
    "Middle Africa",
    "Northern Africa",
    "Northern America",
    "Northern Europe",
    "Polynesia",
    "South America",
    "South-eastern Asia",
    "Southern Africa",
    "Southern Asia",
    "Southern Europe",
    "Western Africa",
    "Western Asia",
    "Western Europe",
]

# Per-group ISIC × ISCO overlap factors derived from ILO WESO 2023
# Figure A1 (Annex). For each occupational group g, ``GROUP_OVERLAP[g]``
# is the *global aggregate* fraction of workers in that ISCO occupational
# group who are also employed in a key (essential) ISIC industry.
#
# These factors are the dominant simplification in this pipeline: the
# correct calculation would intersect each country's ISCO × ISIC
# cross-tabulation, but we lack worker-level microdata, so we assume the
# overlap structure is the same in every country. This is sometimes
# materially wrong (e.g. the "Manual" overlap is much higher in
# agrarian economies because more manual workers are in essential
# agriculture). See the module docstring for the full rationale and
# ``validate_against_ilo`` for the per-country deviation it produces.
#
# Source: ILO 2023, "The value of essential work", Figure A1 (Annex):
#   https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@dcomm/@publ/documents/publication/wcms_871016.pdf
#
# ArmedForces sits outside ILO Figure A1 (the report excludes uniformed
# services from its headline global figures). We retain it with the
# Blueprint Biosecurity placeholder of 0.40 so downstream
# indoor-essential counts include armed forces; treat that figure as
# a low-confidence assumption rather than an ILO-derived number.
GROUP_OVERLAP: Dict[str, float] = {
    "Food": 0.895,
    "Health": 0.819,
    "Retail": 0.876,
    "Security": 0.846,
    "Transport": 0.869,
    "Manual": 0.335,
    "Cleaning": 0.485,
    "Tech": 0.320,
    "ArmedForces": 0.40,
}

# Armed Forces overlap is never calibrated (Blueprint assumption).
ARMED_FORCES_OVERLAP_FIXED = 0.40

# Groups adjusted by per-country scalar ``x`` (all except ArmedForces).
CALIBRATABLE_GROUPS: tuple[str, ...] = tuple(
    g for g in GROUP_OVERLAP if g != "ArmedForces"
)

OVERLAP_COL_PREFIX = "overlap_"


def overlap_column(group: str) -> str:
    """Column name for a group's calibrated overlap in LF / calibration tables."""
    return f"{OVERLAP_COL_PREFIX}{group}"


CALIBRATABLE_OVERLAP_COLUMNS: tuple[str, ...] = tuple(
    overlap_column(g) for g in CALIBRATABLE_GROUPS
)

OVERLAP_SOURCE_ILO = "ilo_calibrated"
OVERLAP_SOURCE_NEIGHBOUR = "neighbour_backfill"
OVERLAP_SOURCE_GLOBAL = "global_fallback"

ESSENTIAL_PCT_TOLERANCE_PP = 0.01

# Labour-force / ILO published names that differ from ``ref_area.label`` in
# ``ILO_ISCO_08_GLB.csv`` (country_converter short names).
EMPLOYMENT_COUNTRY_ALIASES: Dict[str, str] = {}

# Armed-forces ISCO-08 level-2 codes used for diagnostic sub-totals.
ARMED_FORCES_L2 = ("01", "02", "03")

# Subsistence farmers (ISCO-08 L2). They are essential/vital per ILO Table A2
# but treated as 100% outdoor and assumed to live on-site already.
SUBSISTENCE_FARMERS_ISCO_L2 = "63"

# ISCO L2 codes excluded from on-site housing requirements: market-oriented
# skilled agricultural workers (61) and subsistence farmers (63). These codes
# usually represent the operators, managers, and owners of farms.We assume
# they live on-site in their own private housing, such as a single-family home
# on a farm. We do not exclude ISCO code 92 (Agricultural, forestry, and
# fishinq workers) as they may be living in poor quality high-density housing.
# For example, seasonal farm workers in the US had high rates of COVID-19
# infection while living in on-site dorms. They would need safer housing in
# a future pandemic.

ONSITE_HOUSING_EXCLUDED_ISCO_L2 = ("61", "63")

# Per-country neighbour map used to back-fill missing labour-force breakdowns
# via the average of nearby / similar-economy countries. Sourced from the
# notebook's hand-curated ``similar_iso3`` mapping.
SIMILAR_ISO3: Dict[str, list] = {
    "ABW": ["CUW", "SXM", "MHL"],
    "AIA": ["VGB", "TCA", "MAF"],
    "AND": ["LIE", "CYP", "MCO"],
    "ARM": ["GEO", "AZE", "ALB"],
    "ASM": ["GUM", "MNP", "WSM"],
    "ATA": ["ATF", "HMD", "SGS"],
    "ATF": ["HMD", "BVT", "SGS"],
    "ATG": ["KNA", "MDG", "VCT"],
    "AZE": ["GEO", "KAZ", "UZB"],
    "BES": ["ABW", "CUW", "SXM"],
    "BHR": ["ARE", "QAT", "OMN"],
    "BLM": ["MAF", "SXM", "GLP"],
    "BMU": ["MHL"],
    "BVT": ["ATF", "HMD", "SGS"],
    "CAF": ["TCD", "SSD", "NER"],
    "CAN": ["USA"],
    "CCK": ["CXR", "NFK", "HMD"],
    "CHI": ["GBR"],
    "CHN": ["JPN", "IND", "VNM"],
    "CMR": ["COG", "GAB", "NGA"],
    "COG": ["GAB", "CMR", "GNQ"],
    "COM": ["MDG", "MUS", "SYC"],
    "CPV": ["STP", "COM", "MUS"],
    "CUB": ["JAM", "DOM", "PRI"],
    "CUW": ["ABW", "MHL"],
    "CXR": ["CCK", "NFK", "HMD"],
    "CYM": ["VGB", "TCA", "BMU"],
    "DJI": ["ERI", "SOM", "YEM"],
    "DMA": ["KNA", "VCT", "LCA"],
    "DZA": ["MAR", "TUN", "LBY"],
    "ERI": ["DJI", "SOM", "SDN"],
    "ESH": ["MAR", "MRT", "DZA"],
    "FLK": ["SGS", "SHN", "BVT"],
    "FRO": ["ISL", "GRL"],
    "FSM": ["MHL", "KIR", "PLW"],
    "GAB": ["GNQ", "COG", "AGO"],
    "GGY": ["JEY", "IMN", "BMU"],
    "GIB": ["MLT", "AND", "LIE"],
    "GLP": ["MTQ", "MAF", "BLM"],
    "GNQ": ["GAB", "COG", "STP"],
    "GRL": ["ISL", "FRO"],
    "GUF": ["SUR", "GUY", "MTQ"],
    "GUM": ["MNP", "ASM", "PLW"],
    "HKG": ["MAC", "SGP", "CHN"],
    "HMD": ["ATF", "BVT", "SGS"],
    "HTI": ["NIC", "JAM", "HND"],
    "IMN": ["CHI", "GBR"],
    "IOT": ["HMD", "CCK", "CXR"],
    "JAM": ["BRB", "TTO", "BHS"],
    "JEY": ["GGY", "IMN", "BMU"],
    "KAZ": ["UZB", "TKM", "AZE"],
    "KOR": ["JPN", "CHN"],
    "KNA": ["ATG", "DMA", "VCT"],
    "KWT": ["QAT", "BHR", "OMN"],
    "LBY": ["DZA", "TUN", "EGY"],
    "LCA": ["VCT", "DMA", "ATG"],
    "LIE": ["CYP", "SMR", "MCO"],
    "MAC": ["HKG", "SGP", "CHN"],
    "MAF": ["SXM", "MDG"],
    "MAR": ["TUN", "DZA", "EGY"],
    "MCO": ["CYP", "LIE", "SMR"],
    "MDA": ["UKR", "GEO", "ALB"],
    "MLT": ["CYP", "MNE", "ISL"],
    "MNP": ["GUM", "ASM", "PLW"],
    "MRT": ["TCD", "NER"],
    "MTQ": ["GLP", "MAF", "BLM"],
    "MWI": ["MOZ", "ZMB", "TZA"],
    "MYS": ["THA", "IDN", "VNM"],
    "MYT": ["REU", "COM", "MUS"],
    "NCL": ["WSM"],
    "NFK": ["CCK", "CXR", "HMD"],
    "NIC": ["HND", "GTM", "SLV"],
    "NZL": ["AUS"],
    "OMN": ["QAT", "ARE", "BHR"],
    "PCN": ["TKL", "NIU", "NFK"],
    "PRI": ["PAN", "TTO", "JAM"],
    "PRK": ["VNM", "LAO", "MMR"],
    "PRY": ["BOL", "PER", "URY"],
    "PYF": ["WSM"],
    "QAT": ["KWT", "BHR", "ARE"],
    "REU": ["MYT", "MUS", "COM"],
    "SAU": ["ARE", "QAT", "ARE"],
    "SGS": ["FLK", "BVT", "ATF"],
    "SHN": ["FLK", "PCN", "NFK"],
    "SJM": ["GRL", "FRO", "ISL"],
    "SLB": ["VUT", "PNG", "FJI"],
    "SMR": ["CYP", "LIE", "MCO"],
    "SPM": ["BMU", "JEY", "GGY"],
    "SSD": ["TCD", "CAF", "ERI"],
    "SXM": ["ABW", "CUW", "MAF"],
    "SYR": ["IRQ", "JOR", "LBN"],
    "TCA": ["CYM", "VGB", "ABW"],
    "TCD": ["CAF", "NER", "SSD"],
    "TKM": ["UZB", "KAZ", "AZE"],
    "TWN": ["KOR", "JPN", "HKG"],
    "UMI": ["PCN", "NFK", "CXR"],
    "UZB": ["KAZ", "TKM", "KGZ"],
    "VAT": ["SMR", "MCO", "CYP"],
    "VCT": ["LCA", "DMA", "ATG"],
    "VEN": ["COL", "ECU", "PER"],
    "VGB": ["CYM", "TCA", "ABW"],
    "VIR": ["VGB", "ABW", "CUW"],
    "YEM": ["SOM", "SDN", "ERI"],
}


def overlap_calibration_feasible(
    overlap_country_df: pd.DataFrame,
) -> pd.Series:
    """Boolean mask: countries with ILO calibration that reached the target (``ok``)."""
    return overlap_country_df["solver_status"].isin(("ok", "exact_at_baseline"))


def build_isco_lvl2_weights(
    poll_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    onet_controlled_df: Optional[pd.DataFrame] = None,
    onet_not_controlled_df: Optional[pd.DataFrame] = None,
    *,
    indoor_context_method: IndoorContextMethod = "onet_max",
    jem_path: Optional[Path] = None,
    soc_to_isco_aggregator: str = "mean",
) -> pd.DataFrame:
    """Build ``ISCO_LVL2_WEIGHTS`` with global :data:`GROUP_OVERLAP`."""
    template = build_isco_lvl2_template(
        poll_df,
        crosswalk_df,
        onet_controlled_df=onet_controlled_df,
        onet_not_controlled_df=onet_not_controlled_df,
        indoor_context_method=indoor_context_method,
        jem_path=jem_path,
        soc_to_isco_aggregator=soc_to_isco_aggregator,
    )
    return apply_group_overlaps(template, GROUP_OVERLAP)


def apply_group_overlaps(
    lvl2_template: pd.DataFrame,
    group_overlaps: Dict[str, float],
) -> pd.DataFrame:
    """Attach per-group overlaps and compute the four ISCO weight columns."""
    lvl2 = lvl2_template.copy()
    overlaps = dict(group_overlaps)
    overlaps["ArmedForces"] = ARMED_FORCES_OVERLAP_FIXED
    lvl2["Group Overlap"] = lvl2["Group"].map(overlaps).fillna(0.0)

    lvl2["ISCO_08_PollWeights"] = (
        lvl2["Vital Weight POLL"] * lvl2[INDOORS_CONTEXT_COLUMN] * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_ILOWeights"] = (
        lvl2["Essential Weight ILO"]
        * lvl2[INDOORS_CONTEXT_COLUMN]
        * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_PollWeights_Total"] = (
        lvl2["Vital Weight POLL"] * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_ILOWeights_Total"] = (
        lvl2["Essential Weight ILO"] * lvl2["Group Overlap"]
    )
    return lvl2


# ---------------------------------------------------------------------------
# Per-country group overlap calibration (ILO baseline)
# ---------------------------------------------------------------------------


def essential_mass_by_group(
    employment: Dict[str, float],
    weights_template: pd.DataFrame,
) -> Dict[str, float]:
    """Employment in essential ISCO codes, by ILO Figure A1 group (no overlap)."""
    masses = {g: 0.0 for g in GROUP_OVERLAP}
    for code, emp in employment.items():
        code_str = str(code).strip()
        if code_str == "Tot" or not pd.notna(emp):
            continue
        if code_str not in weights_template.index:
            continue
        if weights_template.at[code_str, "Essential Weight ILO"] != 1:
            continue
        group = weights_template.at[code_str, "Group"]
        if group in masses:
            masses[group] += float(emp)
    return masses


def essential_mass_at_overlaps(
    masses_by_group: Dict[str, float],
    group_overlaps: Dict[str, float],
) -> float:
    """Weighted essential employment mass ``Σ_g o_g × S_g``."""
    return sum(
        float(group_overlaps.get(g, 0.0)) * float(masses_by_group.get(g, 0.0))
        for g in GROUP_OVERLAP
    )


@dataclass
class OverlapCalibrationResult:
    """Per-country overlap calibration vs global ``GROUP_OVERLAP``."""

    overlaps_by_country: Dict[str, Dict[str, float]]
    country_table: pd.DataFrame
    detail_df: pd.DataFrame


def calibrate_group_overlaps(
    masses_by_group: Dict[str, float],
    target_essential: float,
    baseline: Optional[Dict[str, float]] = None,
    *,
    tol: float = 1e-6,
) -> tuple[Dict[str, float], float, str, str]:
    """Solve scalar ``x`` so essential mass matches ``target_essential``.

    Uses proportional headroom toward 1 (raise) or toward 0 (lower). Armed
    Forces overlap is fixed at :data:`ARMED_FORCES_OVERLAP_FIXED`.

    Returns ``(overlaps, x, direction, status)`` where ``status`` is
    ``ok``, ``exact_at_baseline``, or ``infeasible_clipped``.
    """
    baseline = dict(baseline or GROUP_OVERLAP)
    o0 = {g: float(baseline[g]) for g in GROUP_OVERLAP}
    o0["ArmedForces"] = ARMED_FORCES_OVERLAP_FIXED

    e0 = essential_mass_at_overlaps(masses_by_group, o0)
    target = float(target_essential)

    if target <= 0:
        overlaps = {
            g: 0.0 if g != "ArmedForces" else ARMED_FORCES_OVERLAP_FIXED
            for g in GROUP_OVERLAP
        }
        return overlaps, 1.0, "lower", "ok"

    if abs(e0 - target) <= tol * max(target, 1.0):
        return dict(o0), 0.0, "none", "exact_at_baseline"

    if e0 < target:
        direction = "raise"
        denom = sum(
            (1.0 - o0[g]) * masses_by_group.get(g, 0.0) for g in CALIBRATABLE_GROUPS
        )
        x_raw = 1.0 if denom <= 0 else (target - e0) / denom
        x = float(np.clip(x_raw, 0.0, 1.0))
        overlaps = {
            g: (
                ARMED_FORCES_OVERLAP_FIXED
                if g == "ArmedForces"
                else o0[g] + x * (1.0 - o0[g])
            )
            for g in GROUP_OVERLAP
        }
    else:
        direction = "lower"
        denom = sum(o0[g] * masses_by_group.get(g, 0.0) for g in CALIBRATABLE_GROUPS)
        x_raw = 1.0 if denom <= 0 else (e0 - target) / denom
        x = float(np.clip(x_raw, 0.0, 1.0))
        overlaps = {
            g: (ARMED_FORCES_OVERLAP_FIXED if g == "ArmedForces" else o0[g] * (1.0 - x))
            for g in GROUP_OVERLAP
        }

    e1 = essential_mass_at_overlaps(masses_by_group, overlaps)
    if abs(x_raw - x) > 1e-5 or abs(e1 - target) > tol * max(target, 1.0):
        status = "infeasible_clipped"
    else:
        status = "ok"
    return overlaps, x, direction, status


def _ilo_target_essential(
    tot_employment: float,
    ilo_pct_essential: float,
) -> float:
    return (ilo_pct_essential / 100.0) * tot_employment


def calibrate_overlaps_for_country(
    country: str,
    employment: Dict[str, float],
    weights_template: pd.DataFrame,
    ilo_pct_essential: float,
) -> tuple[Dict[str, float], dict]:
    """Calibrate overlaps for one country with ILO microdata and published %."""
    tot = employment.get("Tot")
    if not tot or not pd.notna(tot) or tot <= 0:
        raise ValueError(f"{country}: invalid Tot employment")
    masses = essential_mass_by_group(employment, weights_template)
    target = _ilo_target_essential(tot, ilo_pct_essential)
    overlaps, x, direction, status = calibrate_group_overlaps(masses, target)
    e0 = essential_mass_at_overlaps(masses, GROUP_OVERLAP)
    meta = {
        "calibration_x": x,
        "calibration_direction": direction,
        "solver_status": status,
        "model_essential_mass": e0,
        "ilo_target_mass": target,
        "overlap_source": OVERLAP_SOURCE_ILO,
    }
    return overlaps, meta


def build_overlap_country_table(
    lf_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    ilo_pct_df: pd.DataFrame,
    weights_template: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, dict]]:
    """Calibrate overlaps for countries with ILO emp + published %; NaN otherwise."""
    ilo_lookup = ilo_pct_df.set_index("Country Name")[
        "ILO %essential (published)"
    ].to_dict()
    rows = []
    overlaps_by_country: Dict[str, Dict[str, float]] = {}
    meta_by_country: Dict[str, dict] = {}

    for _, lf_row in lf_df.iterrows():
        country = lf_row["Country Name"]
        code = lf_row["Country Code"]
        row = {
            "Country Name": country,
            "Country Code": code,
        }
        for col in CALIBRATABLE_OVERLAP_COLUMNS:
            row[col] = np.nan
        row["calibration_x"] = np.nan
        row["calibration_direction"] = ""
        row["solver_status"] = ""
        row["overlap_source"] = ""

        emp = employment_for_country(
            country, employment_by_iso, EMPLOYMENT_COUNTRY_ALIASES
        )
        ilo_pct = ilo_lookup.get(country)
        if emp and ilo_pct is not None and pd.notna(ilo_pct):
            tot = emp.get("Tot")
            if tot and pd.notna(tot) and tot > 0:
                overlaps, meta = calibrate_overlaps_for_country(
                    country, emp, weights_template, float(ilo_pct)
                )
                overlaps_by_country[country] = overlaps
                meta_by_country[country] = meta
                for g in CALIBRATABLE_GROUPS:
                    row[overlap_column(g)] = overlaps[g]
                row["calibration_x"] = meta["calibration_x"]
                row["calibration_direction"] = meta["calibration_direction"]
                row["solver_status"] = meta["solver_status"]
                row["overlap_source"] = meta["overlap_source"]
                row["model_essential_mass"] = meta["model_essential_mass"]
                row["ilo_target_mass"] = meta["ilo_target_mass"]

        rows.append(row)

    return pd.DataFrame(rows), overlaps_by_country, meta_by_country


def backfill_calibrated_overlaps(
    overlap_df: pd.DataFrame,
    similar_iso3: Optional[Dict[str, list]] = None,
    *,
    max_iterations: int = 50,
) -> pd.DataFrame:
    """Fill NaN group overlaps from SIMILAR_ISO3 neighbours' calibrated values."""
    if similar_iso3 is None:
        similar_iso3 = SIMILAR_ISO3
    df = overlap_df.copy()

    for _ in range(max_iterations):
        still_missing = False
        for idx, row in df.iterrows():
            if not pd.isna(row.get(overlap_column(CALIBRATABLE_GROUPS[0]))):
                continue
            neighbours = similar_iso3.get(row["Country Code"], [])
            for col in CALIBRATABLE_OVERLAP_COLUMNS:
                values = []
                for iso in neighbours:
                    n_row = df.loc[df["Country Code"] == iso]
                    if n_row.empty:
                        continue
                    if n_row.iloc[0]["overlap_source"] not in (
                        OVERLAP_SOURCE_ILO,
                        OVERLAP_SOURCE_NEIGHBOUR,
                    ):
                        continue
                    v = n_row.iloc[0][col]
                    if pd.notna(v):
                        values.append(float(v))
                if values:
                    df.at[idx, col] = sum(values) / len(values)
                else:
                    still_missing = True
            if not pd.isna(df.at[idx, overlap_column(CALIBRATABLE_GROUPS[0])]):
                sources = []
                for iso in neighbours:
                    n_row = df.loc[df["Country Code"] == iso]
                    if (
                        not n_row.empty
                        and n_row.iloc[0]["overlap_source"] == OVERLAP_SOURCE_ILO
                    ):
                        sources.append(iso)
                if sources:
                    df.at[idx, "overlap_source"] = OVERLAP_SOURCE_NEIGHBOUR
                x_vals = [
                    float(df.loc[df["Country Code"] == iso, "calibration_x"].iloc[0])
                    for iso in neighbours
                    if not df.loc[df["Country Code"] == iso, "calibration_x"].empty
                    and pd.notna(
                        df.loc[df["Country Code"] == iso, "calibration_x"].iloc[0]
                    )
                ]
                if x_vals:
                    df.at[idx, "calibration_x"] = sum(x_vals) / len(x_vals)
        if not still_missing:
            break

    for idx, row in df.iterrows():
        if pd.isna(row.get(overlap_column(CALIBRATABLE_GROUPS[0]))):
            for g in CALIBRATABLE_GROUPS:
                df.at[idx, overlap_column(g)] = GROUP_OVERLAP[g]
            df.at[idx, "overlap_source"] = OVERLAP_SOURCE_GLOBAL
            df.at[idx, "solver_status"] = "global_fallback"
            df.at[idx, "calibration_x"] = 0.0

    return df


def overlap_df_to_dict(row: pd.Series) -> Dict[str, float]:
    """Row with ``overlap_*`` columns → group overlap dict including ArmedForces."""
    out = dict(GROUP_OVERLAP)
    for g in CALIBRATABLE_GROUPS:
        col = overlap_column(g)
        if col in row.index and pd.notna(row[col]):
            out[g] = float(row[col])
    out["ArmedForces"] = ARMED_FORCES_OVERLAP_FIXED
    return out


def build_group_overlap_calibration_detail(
    lf_df: pd.DataFrame,
    overlap_country_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    weights_template: pd.DataFrame,
    ilo_pct_df: pd.DataFrame,
    workers_model: WorkerDicts,
    workers_calibrated: WorkerDicts,
) -> pd.DataFrame:
    """Long-format table for paper / diagnostics (country × group)."""
    ilo_lookup = ilo_pct_df.set_index("Country Name")[
        "ILO %essential (published)"
    ].to_dict()
    records = []
    for _, orow in overlap_country_df.iterrows():
        country = orow["Country Name"]
        emp = (
            employment_for_country(
                country, employment_by_iso, EMPLOYMENT_COUNTRY_ALIASES
            )
            or {}
        )
        masses = essential_mass_by_group(emp, weights_template) if emp else {}
        model_pct = workers_model.ew_pc.get(country)
        cal_pct = workers_calibrated.ew_pc.get(country)
        ilo_pct = ilo_lookup.get(country)
        for g in GROUP_OVERLAP:
            o0 = GROUP_OVERLAP[g]
            if g == "ArmedForces":
                og = ARMED_FORCES_OVERLAP_FIXED
            else:
                og = float(orow.get(overlap_column(g), np.nan))
                if pd.isna(og):
                    og = o0
            records.append(
                {
                    "Country Name": country,
                    "Country Code": orow["Country Code"],
                    "Group": g,
                    "Global overlap": o0,
                    "Calibrated overlap": og,
                    "Adjustment": og - o0,
                    "Group essential mass S_g": masses.get(g, np.nan),
                    "Overlap source": orow.get("overlap_source", ""),
                    "calibration_x": orow.get("calibration_x", np.nan),
                    "ILO %essential (published)": ilo_pct,
                    "Model %Essential (pct)": (
                        100 * model_pct
                        if model_pct is not None and pd.notna(model_pct)
                        else np.nan
                    ),
                    "Calibrated %Essential (pct)": (
                        100 * cal_pct
                        if cal_pct is not None and pd.notna(cal_pct)
                        else np.nan
                    ),
                    "Delta model (pp)": (
                        100 * model_pct - ilo_pct
                        if model_pct is not None
                        and pd.notna(model_pct)
                        and ilo_pct is not None
                        and pd.notna(ilo_pct)
                        else np.nan
                    ),
                    "Delta calibrated (pp)": (
                        100 * cal_pct - ilo_pct
                        if cal_pct is not None
                        and pd.notna(cal_pct)
                        and ilo_pct is not None
                        and pd.notna(ilo_pct)
                        else np.nan
                    ),
                    "solver_status": orow.get("solver_status", ""),
                }
            )
    return pd.DataFrame(records)


def calibrate_country_overlaps(
    lf_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    ilo_pct_df: pd.DataFrame,
    weights_template: pd.DataFrame,
    workers_model: Optional["WorkerDicts"] = None,
    workers_calibrated: Optional["WorkerDicts"] = None,
) -> OverlapCalibrationResult:
    """Full overlap calibration + neighbour back-fill for all LF countries."""
    country_df, _overlaps_ilo, _meta = build_overlap_country_table(
        lf_df, employment_by_iso, ilo_pct_df, weights_template
    )
    country_df = backfill_calibrated_overlaps(country_df)
    overlaps_by_country: Dict[str, Dict[str, float]] = {}
    for _, row in country_df.iterrows():
        overlaps_by_country[row["Country Name"]] = overlap_df_to_dict(row)

    detail_df = pd.DataFrame()
    if workers_model is not None and workers_calibrated is not None:
        detail_df = build_group_overlap_calibration_detail(
            lf_df,
            country_df,
            employment_by_iso,
            weights_template,
            ilo_pct_df,
            workers_model,
            workers_calibrated,
        )

    return OverlapCalibrationResult(
        overlaps_by_country=overlaps_by_country,
        country_table=country_df,
        detail_df=detail_df,
    )


def build_dual_validation_merged(
    lf_df: pd.DataFrame,
    ilo_pct_df: pd.DataFrame,
    workers_model: WorkerDicts,
    validation_calibrated: ValidationResult,
) -> pd.DataFrame:
    """Merge calibrated validation with pre-calibration (global overlap) %Essential."""
    merged = validation_calibrated.merged_df.copy()
    merged["Our %Essential (model, global overlap)"] = merged["Country Name"].map(
        lambda c: (
            100.0 * workers_model.ew_pc[c]
            if c in workers_model.ew_pc and pd.notna(workers_model.ew_pc[c])
            else np.nan
        )
    )
    merged["Delta model (pp)"] = (
        merged["Our %Essential (model, global overlap)"]
        - merged["ILO %essential (published)"]
    )
    merged = merged.rename(
        columns={
            "Our %Essential (pct)": "Our %Essential (calibrated)",
            "Delta (pp)": "Delta calibrated (pp)",
        }
    )
    return merged


@dataclass
class WorkerDicts:
    """Bundle of per-country worker counts and percentages."""

    iew_ilo: Dict[str, float] = field(
        default_factory=dict
    )  # indoor essential workers (ilo)
    ew_ilo: Dict[str, float] = field(default_factory=dict)  # essential workers (ilo)
    ivw_poll: Dict[str, float] = field(
        default_factory=dict
    )  # indoor vital workers (poll)
    vw_poll: Dict[str, float] = field(default_factory=dict)  # vital workers (poll)
    af_indoor_essential: Dict[str, float] = field(
        default_factory=dict
    )  # armed forces (indoor essential)
    af_essential: Dict[str, float] = field(
        default_factory=dict
    )  # armed forces (essential)
    iew_pc: Dict[str, float] = field(
        default_factory=dict
    )  # indoor essential workers (percentage)
    ew_pc: Dict[str, float] = field(
        default_factory=dict
    )  # essential workers (percentage)
    ivw_pc: Dict[str, float] = field(
        default_factory=dict
    )  # indoor vital workers (percentage)
    vw_pc: Dict[str, float] = field(default_factory=dict)  # vital workers (percentage)
    af_indoor_essential_pc: Dict[str, float] = field(
        default_factory=dict
    )  # armed forces (indoor essential) (percentage)
    af_essential_pc: Dict[str, float] = field(
        default_factory=dict
    )  # armed forces (essential) (percentage)


def compute_worker_dicts(
    employment_by_iso: Dict[str, Dict[str, float]],
    weights_template: pd.DataFrame,
    overlaps_by_country: Optional[Dict[str, Dict[str, float]]] = None,
) -> WorkerDicts:
    """Compute per-country indoor / total essential & vital worker counts.

    When ``overlaps_by_country`` is provided, each country's group overlaps
    are applied via :func:`apply_group_overlaps` before summing employment.
    Otherwise global :data:`GROUP_OVERLAP` is used for every country.
    """
    if overlaps_by_country is None:
        overlaps_by_country = {c: dict(GROUP_OVERLAP) for c in employment_by_iso}

    out = WorkerDicts()

    for country, code_dict in employment_by_iso.items():
        overlaps = overlaps_by_country.get(country, GROUP_OVERLAP)
        weights = apply_group_overlaps(weights_template, overlaps)
        poll_w = weights["ISCO_08_PollWeights"].to_dict()
        ilo_w = weights["ISCO_08_ILOWeights"].to_dict()
        poll_w_total = weights["ISCO_08_PollWeights_Total"].to_dict()
        ilo_w_total = weights["ISCO_08_ILOWeights_Total"].to_dict()
        iew = ew = ivw = vw = 0.0
        af_ind_e = af_e = 0.0
        coded_emp = 0.0
        for code, employment in code_dict.items():
            code_str = str(code).strip()
            if not pd.notna(employment) or code_str in ("Tot", "Not"):
                continue
            coded_emp += employment
            if code_str in poll_w:
                w = poll_w[code_str]
                ivw += employment * w if pd.notna(w) else 0
                wt = poll_w_total[code_str]
                vw += employment * wt if pd.notna(wt) else 0
            if code_str in ilo_w:
                w = ilo_w[code_str]
                contrib_indoor = employment * w if pd.notna(w) else 0
                iew += contrib_indoor
                wt = ilo_w_total[code_str]
                contrib_total = employment * wt if pd.notna(wt) else 0
                ew += contrib_total
                if code_str in ARMED_FORCES_L2:
                    af_ind_e += contrib_indoor
                    af_e += contrib_total

        nec_emp = code_dict.get("Not")
        if nec_emp is not None and pd.notna(nec_emp) and nec_emp > 0 and coded_emp > 0:
            avg_ew = ew / coded_emp
            avg_vw = vw / coded_emp
            avg_iew = iew / coded_emp
            avg_ivw = ivw / coded_emp
            ew += nec_emp * avg_ew
            vw += nec_emp * avg_vw
            iew += nec_emp * avg_iew
            ivw += nec_emp * avg_ivw

        out.iew_ilo[country] = iew
        out.ew_ilo[country] = ew
        out.ivw_poll[country] = ivw
        out.vw_poll[country] = vw
        out.af_indoor_essential[country] = af_ind_e
        out.af_essential[country] = af_e

        total_employment = code_dict.get("Tot")
        if total_employment and total_employment > 0:
            out.iew_pc[country] = iew / total_employment
            out.ew_pc[country] = ew / total_employment
            out.ivw_pc[country] = ivw / total_employment
            out.vw_pc[country] = vw / total_employment
            out.af_indoor_essential_pc[country] = af_ind_e / total_employment
            out.af_essential_pc[country] = af_e / total_employment

    return out


def _employment_in_codes(
    emp: Dict[str, float],
    codes: Iterable[str],
    weights_by_code: Optional[Dict[str, float]] = None,
) -> float:
    """Sum employment for ISCO L2 codes, optionally multiplied by per-code weights."""
    total = 0.0
    for code in codes:
        w = 1.0 if weights_by_code is None else float(weights_by_code.get(code, 0.0))
        for k, n in emp.items():
            if str(k).strip() == code and pd.notna(n):
                total += float(n) * w
                break
    return total


def _onsite_excluded_weighted_employment_pct(
    employment_by_iso: Dict[str, Dict[str, float]],
    country: str,
    weights: pd.DataFrame,
    weight_column: str,
    codes: Iterable[str] = ONSITE_HOUSING_EXCLUDED_ISCO_L2,
) -> float:
    """Share of ILO employment in excluded codes, weighted for one worker series."""
    emp = employment_by_iso.get(country)
    if not emp:
        return np.nan
    tot = emp.get("Tot")
    if not tot or not pd.notna(tot) or tot <= 0:
        return np.nan
    w_by_code = weights[weight_column].to_dict()
    return _employment_in_codes(emp, codes, w_by_code) / tot


def _country_weights_for_onsite(
    weights_template: pd.DataFrame,
    country: str,
    overlaps_by_country: Optional[Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """Per-country weight table for on-site excluded shares."""
    if (
        overlaps_by_country is None
        and "ISCO_08_ILOWeights_Total" in weights_template.columns
    ):
        return weights_template
    overlaps = (overlaps_by_country or {}).get(country, GROUP_OVERLAP)
    return apply_group_overlaps(weights_template, overlaps)


def onsite_excluded_essential_employment_pct(
    employment_by_iso: Dict[str, Dict[str, float]],
    country: str,
    weights_template: pd.DataFrame,
    overlaps_by_country: Optional[Dict[str, Dict[str, float]]] = None,
    codes: Iterable[str] = ONSITE_HOUSING_EXCLUDED_ISCO_L2,
) -> float:
    """Share of employment contributing to total essential for excluded codes.

    Weighted by ``ISCO_08_ILOWeights_Total``, matching :func:`compute_worker_dicts`
    / ``%Essential Workers``.
    """
    weights = _country_weights_for_onsite(
        weights_template, country, overlaps_by_country
    )
    return _onsite_excluded_weighted_employment_pct(
        employment_by_iso, country, weights, "ISCO_08_ILOWeights_Total", codes
    )


def onsite_excluded_vital_employment_pct(
    employment_by_iso: Dict[str, Dict[str, float]],
    country: str,
    weights_template: pd.DataFrame,
    overlaps_by_country: Optional[Dict[str, Dict[str, float]]] = None,
    codes: Iterable[str] = ONSITE_HOUSING_EXCLUDED_ISCO_L2,
) -> float:
    """Share of employment contributing to total vital for excluded codes.

    Weighted by ``ISCO_08_PollWeights_Total``, matching :func:`compute_worker_dicts`
    / ``%Vital Workers``.
    """
    weights = _country_weights_for_onsite(
        weights_template, country, overlaps_by_country
    )
    return _onsite_excluded_weighted_employment_pct(
        employment_by_iso, country, weights, "ISCO_08_PollWeights_Total", codes
    )


ONSITE_EXCLUDED_ESSENTIAL_PCT_COL = "%Onsite Excluded Essential (ISCO 61+63)"
ONSITE_EXCLUDED_VITAL_PCT_COL = "%Onsite Excluded Vital (ISCO 61+63)"


def attach_onsite_excluded_pct(
    lf_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    weights_template: pd.DataFrame,
    overlaps_by_country: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Attach ILO- and poll-weighted excluded shares for essential and vital."""
    df = lf_df.copy()
    df[ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] = df["Country Name"].map(
        lambda c: onsite_excluded_essential_employment_pct(
            employment_by_iso, c, weights_template, overlaps_by_country
        )
    )
    df[ONSITE_EXCLUDED_VITAL_PCT_COL] = df["Country Name"].map(
        lambda c: onsite_excluded_vital_employment_pct(
            employment_by_iso,
            c,
            weights_template,
            overlaps_by_country,
        )
    )
    return df


def build_onsite_housing_worker_requirements(
    lf_df: pd.DataFrame,
    lf_col: str = "Labour Force (2024)",
    excluded_essential_col: str = ONSITE_EXCLUDED_ESSENTIAL_PCT_COL,
    excluded_vital_col: str = ONSITE_EXCLUDED_VITAL_PCT_COL,
) -> pd.DataFrame:
    """Essential/vital totals minus LF-scaled on-site-excluded ISCO 61 and 63.

    Essential and vital each subtract only the slice counted in their totals:
    ``ISCO_08_ILOWeights_Total`` and ``ISCO_08_PollWeights_Total`` respectively.
    """
    excluded_essential = lf_df[excluded_essential_col].fillna(0.0) * lf_df[lf_col]
    excluded_vital = lf_df[excluded_vital_col].fillna(0.0) * lf_df[lf_col]

    out = lf_df[
        ["Country Name", "Country Code", "Essential Workers", "Vital Workers"]
    ].copy()
    out["Essential Workers (Housing Requirement)"] = (
        lf_df["Essential Workers"] - excluded_essential
    )
    out["Vital Workers (Housing Requirement)"] = lf_df["Vital Workers"] - excluded_vital

    global_row = {
        "Country Name": "Global",
        "Country Code": "GLOBAL",
        "Essential Workers (Housing Requirement)": out[
            "Essential Workers (Housing Requirement)"
        ].sum(skipna=True),
        "Vital Workers (Housing Requirement)": out[
            "Vital Workers (Housing Requirement)"
        ].sum(skipna=True),
        "Essential Workers": lf_df["Essential Workers"].sum(skipna=True),
        "Vital Workers": lf_df["Vital Workers"].sum(skipna=True),
    }
    return pd.concat([pd.DataFrame([global_row]), out], ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Labour force join, back-fill and absolute counts
# ---------------------------------------------------------------------------


_PCT_COLUMNS = [
    "%Indoor Essential Workers",
    "%Indoor Vital Workers",
    "%Essential Workers",
    "%Vital Workers",
    "%Armed Forces (Indoor Essential)",
    "%Armed Forces (Essential)",
]
_BACKFILL_COLUMNS = [
    "%Indoor Essential Workers",
    "%Essential Workers",
    "%Indoor Vital Workers",
    "%Vital Workers",
]
_COUNT_COLUMNS = [
    ("Indoor Essential Workers", "%Indoor Essential Workers"),
    ("Indoor Vital Workers", "%Indoor Vital Workers"),
    ("Essential Workers", "%Essential Workers"),
    ("Vital Workers", "%Vital Workers"),
    ("Armed Forces (Indoor Essential)", "%Armed Forces (Indoor Essential)"),
    ("Armed Forces (Essential)", "%Armed Forces (Essential)"),
]

ONSITE_HOUSING_WORKER_COUNT_COLUMNS: tuple[str, ...] = (
    "Essential Workers (Housing Requirement)",
    "Vital Workers (Housing Requirement)",
    "Essential Workers",
    "Vital Workers",
)


def attach_pct_columns(lf_df: pd.DataFrame, workers: WorkerDicts) -> pd.DataFrame:
    """Attach the six per-country percentage columns from a ``WorkerDicts``."""
    df = lf_df.copy()
    for col in _PCT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        country = row["Country Name"]
        if country in workers.iew_pc:
            df.at[idx, "%Indoor Essential Workers"] = workers.iew_pc[country]
            df.at[idx, "%Indoor Vital Workers"] = workers.ivw_pc[country]
            df.at[idx, "%Essential Workers"] = workers.ew_pc[country]
            df.at[idx, "%Vital Workers"] = workers.vw_pc[country]
        if country in workers.af_indoor_essential_pc:
            df.at[idx, "%Armed Forces (Indoor Essential)"] = (
                workers.af_indoor_essential_pc[country]
            )
            df.at[idx, "%Armed Forces (Essential)"] = workers.af_essential_pc[country]
    return df


def backfill_neighbours(
    lf_df: pd.DataFrame,
    similar_iso3: Optional[Dict[str, list]] = None,
    cols: Iterable[str] = _BACKFILL_COLUMNS,
    max_iterations: int = 50,
) -> pd.DataFrame:
    """Fill missing percentages from the average of neighbour ISO-3 countries.

    The fill is iterated because a country's neighbours may themselves be
    filled in a later sweep. ``max_iterations`` guards against pathological
    inputs - the real data converges in just a few passes.
    """
    if similar_iso3 is None:
        similar_iso3 = SIMILAR_ISO3
    df = lf_df.copy()
    for col in cols:
        for _ in range(max_iterations):
            still_missing = False
            for idx, row in df.iterrows():
                if not pd.isna(row[col]):
                    continue
                code = row["Country Code"]
                neighbours = similar_iso3.get(code, [])
                values = []
                for iso in neighbours:
                    match = df.loc[df["Country Code"] == iso, col]
                    if not match.empty:
                        v = match.iloc[0]
                        if not pd.isna(v):
                            values.append(float(v))
                if values:
                    df.at[idx, col] = sum(values) / len(values)
                else:
                    still_missing = True
            if not still_missing:
                break
    return df


def fill_missing_labour_force_from_ilo_tot(
    lf_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    lf_col: str = "Labour Force (2024)",
    employment_country_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fill missing World Bank labour force with ILO ``Tot`` employment (persons).

    ``build_employment_by_isco`` already converts ILO thousands to persons. Used when
    ``LFData_WB_plus.xlsx`` has no figure (e.g. State of Palestine).
    """
    if employment_country_aliases is None:
        employment_country_aliases = EMPLOYMENT_COUNTRY_ALIASES
    df = lf_df.copy()
    for idx, row in df.iterrows():
        if pd.notna(row.get(lf_col)):
            continue
        emp = employment_for_country(
            row["Country Name"],
            employment_by_iso,
            employment_country_aliases,
        )
        if not emp:
            continue
        tot = emp.get("Tot")
        if tot is not None and pd.notna(tot) and tot > 0:
            df.at[idx, lf_col] = float(tot)
    return df


def compute_absolute_counts(
    lf_df: pd.DataFrame, lf_col: str = "Labour Force (2024)"
) -> pd.DataFrame:
    """Multiply each percentage column by the labour force to get counts."""
    df = lf_df.copy()
    for count_col, pct_col in _COUNT_COLUMNS:
        df[count_col] = df[pct_col] * df[lf_col]
    return df


ASHRAE_SCALE_FACTOR = 5.7
SCALED_ECA_COL = "Scaled ECA (L/s/person)"
INDOOR_ESSENTIAL_CADR_COL = "Indoor Essential CADR Requirement (L/s)"
INDOOR_VITAL_CADR_COL = "Indoor Vital CADR Requirement (L/s)"
SCALED_ECA_ESSENTIAL_COL = "Scaled ECA Essential (L/s/person)"
SCALED_ECA_VITAL_COL = "Scaled ECA Vital (L/s/person)"
_CADR_BACKFILL_COLUMNS = (
    SCALED_ECA_ESSENTIAL_COL,
    SCALED_ECA_VITAL_COL,
    INDOOR_ESSENTIAL_CADR_COL,
    INDOOR_VITAL_CADR_COL,
)


def compute_group_workers_and_cadr(
    data_dir: Path,
    lf_df: pd.DataFrame,
    employment_by_iso: Dict[str, Dict[str, float]],
    weights_template: pd.DataFrame,
    overlaps_by_country: Dict[str, Dict[str, float]],
    *,
    scale_factor: float = ASHRAE_SCALE_FACTOR,
    lf_col: str = "Labour Force (2024)",
) -> pd.DataFrame:
    """Per-country occupational-group worker counts and ASHRAE-241 CADR demand."""
    data_dir = Path(data_dir)
    ashrae = pd.read_csv(data_dir / "ASHRAE241_ECA_by_occupancy.csv")
    mapping = pd.read_csv(data_dir / "ASHRAE241_group_mapping.csv")
    mapped = mapping.merge(
        ashrae,
        on=["occupancy_group", "occupancy_category"],
        how="left",
    )
    missing_groups = set(GROUP_OVERLAP) - set(mapped["occupational_group"])
    if missing_groups:
        raise ValueError(f"Missing ASHRAE mapping for groups: {sorted(missing_groups)}")
    if mapped["eca_ls_per_person"].isna().any():
        raise ValueError("Group mapping references unknown ASHRAE occupancy categories")

    group_meta = mapped.set_index("occupational_group")
    rows: list[dict[str, Any]] = []

    for _, lf_row in lf_df.iterrows():
        country = lf_row["Country Name"]
        emp = employment_for_country(country, employment_by_iso)
        if not emp:
            continue
        tot = emp.get("Tot")
        lf = lf_row.get(lf_col)
        if (
            tot is None
            or not pd.notna(tot)
            or tot <= 0
            or lf is None
            or not pd.notna(lf)
            or lf <= 0
        ):
            continue

        overlaps = overlaps_by_country.get(country, GROUP_OVERLAP)
        weights = apply_group_overlaps(weights_template, overlaps)
        poll_w = weights["ISCO_08_PollWeights"].to_dict()
        ilo_w = weights["ISCO_08_ILOWeights"].to_dict()
        poll_w_total = weights["ISCO_08_PollWeights_Total"].to_dict()
        ilo_w_total = weights["ISCO_08_ILOWeights_Total"].to_dict()
        code_to_group = weights["Group"].to_dict()

        group_iew = {g: 0.0 for g in GROUP_OVERLAP}
        group_ivw = {g: 0.0 for g in GROUP_OVERLAP}
        group_ew = {g: 0.0 for g in GROUP_OVERLAP}
        group_vw = {g: 0.0 for g in GROUP_OVERLAP}
        coded_emp = 0.0
        iew_coded = ivw_coded = ew_coded = vw_coded = 0.0

        for code, employment in emp.items():
            code_str = str(code).strip()
            if not pd.notna(employment) or code_str in ("Tot", "Not"):
                continue
            coded_emp += employment
            group = code_to_group.get(code_str)
            if group not in GROUP_OVERLAP:
                continue
            if code_str in poll_w:
                w = poll_w[code_str]
                contrib = employment * w if pd.notna(w) else 0.0
                group_ivw[group] += contrib
                ivw_coded += contrib
            if code_str in ilo_w:
                w = ilo_w[code_str]
                contrib = employment * w if pd.notna(w) else 0.0
                group_iew[group] += contrib
                iew_coded += contrib
            if code_str in poll_w_total:
                w = poll_w_total[code_str]
                contrib = employment * w if pd.notna(w) else 0.0
                group_vw[group] += contrib
                vw_coded += contrib
            if code_str in ilo_w_total:
                w = ilo_w_total[code_str]
                contrib = employment * w if pd.notna(w) else 0.0
                group_ew[group] += contrib
                ew_coded += contrib

        nec_emp = emp.get("Not")
        if nec_emp is not None and pd.notna(nec_emp) and nec_emp > 0 and coded_emp > 0:
            avg_iew = iew_coded / coded_emp
            avg_ivw = ivw_coded / coded_emp
            avg_ew = ew_coded / coded_emp
            avg_vw = vw_coded / coded_emp
            for group in GROUP_OVERLAP:
                if iew_coded > 0:
                    group_iew[group] += (
                        nec_emp * avg_iew * (group_iew[group] / iew_coded)
                    )
                if ivw_coded > 0:
                    group_ivw[group] += (
                        nec_emp * avg_ivw * (group_ivw[group] / ivw_coded)
                    )
                if ew_coded > 0:
                    group_ew[group] += nec_emp * avg_ew * (group_ew[group] / ew_coded)
                if vw_coded > 0:
                    group_vw[group] += nec_emp * avg_vw * (group_vw[group] / vw_coded)

        for group in GROUP_OVERLAP:
            meta = group_meta.loc[group]
            scaled_eca = float(meta["eca_ls_per_person"]) * scale_factor
            scale = float(lf) / tot
            indoor_essential = group_iew[group] * scale
            indoor_vital = group_ivw[group] * scale
            essential = group_ew[group] * scale
            vital = group_vw[group] * scale
            rows.append(
                {
                    "Country Name": country,
                    "Country Code": lf_row["Country Code"],
                    "Region": lf_row["Region"],
                    "occupational_group": group,
                    "occupancy_group": meta["occupancy_group"],
                    "occupancy_category": meta["occupancy_category"],
                    "Essential Workers": essential,
                    "Vital Workers": vital,
                    "Indoor Essential Workers": indoor_essential,
                    "Indoor Vital Workers": indoor_vital,
                    SCALED_ECA_COL: scaled_eca,
                    INDOOR_ESSENTIAL_CADR_COL: indoor_essential * scaled_eca,
                    INDOOR_VITAL_CADR_COL: indoor_vital * scaled_eca,
                }
            )

    return pd.DataFrame(rows)


def summarize_group_indoor_range_compression(
    group_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    *,
    lf_col: str = "Labour Force (2024)",
) -> pd.DataFrame:
    """Per occupational group, compare country share ranges total vs indoor.

    Uses only countries present in ``group_df`` (those with ILO ISCO employment).
    Shares are group worker counts divided by that country's labour force,
    expressed as percent. A large negative ``% change in range`` for Food
    (and a large total range) supports outdoor/agricultural Food employment
    driving cross-country dispersion in total essential/vital shares.
    """
    if group_df.empty:
        return pd.DataFrame()

    lf_lookup = lf_df.set_index("Country Code")[lf_col]
    merged = group_df.merge(
        lf_lookup.rename("Labour Force").reset_index(),
        on="Country Code",
        how="inner",
    )
    merged = merged[merged["Labour Force"] > 0].copy()

    pairs = (
        ("Essential", "Essential Workers", "Indoor Essential Workers"),
        ("Vital", "Vital Workers", "Indoor Vital Workers"),
    )
    rows: list[dict[str, Any]] = []
    for group, gdf in merged.groupby("occupational_group", sort=True):
        row: dict[str, Any] = {"occupational_group": group}
        for series_label, total_col, indoor_col in pairs:
            total_pct = 100.0 * gdf[total_col] / gdf["Labour Force"]
            indoor_pct = 100.0 * gdf[indoor_col] / gdf["Labour Force"]
            outdoor_pct = total_pct - indoor_pct
            total_range = float(total_pct.max() - total_pct.min())
            indoor_range = float(indoor_pct.max() - indoor_pct.min())
            delta = indoor_range - total_range
            rel = (delta / total_range * 100.0) if total_range else float("nan")
            pm = pitman_morgan_variance_test(total_pct, indoor_pct)
            rel_spread = paired_relative_spread_stats(total_pct, indoor_pct)
            row[f"{series_label} total range (pp)"] = total_range
            row[f"{series_label} indoor range (pp)"] = indoor_range
            row[f"{series_label} Δ range (pp)"] = delta
            row[f"{series_label} % change in range"] = rel
            row[f"{series_label} total SD (pp)"] = pm["sd_x"]
            row[f"{series_label} indoor SD (pp)"] = pm["sd_y"]
            row[f"{series_label} total CV"] = rel_spread["cv_x"]
            row[f"{series_label} indoor CV"] = rel_spread["cv_y"]
            row[f"{series_label} CV ratio (indoor/total)"] = rel_spread[
                "cv_ratio_y_over_x"
            ]
            row[f"{series_label} p (indoor SD < total)"] = pm["p_y_smaller"]
            row[f"{series_label} p (indoor log-SD < total)"] = rel_spread[
                "p_log_y_smaller"
            ]
            row[f"{series_label} mean outdoor % of LF"] = float(outdoor_pct.mean())
        rows.append(row)

    out = pd.DataFrame(rows).set_index("occupational_group")
    return out.sort_values("Essential Δ range (pp)", ascending=True)


# ---------------------------------------------------------------------------
# Country CADR attachment
# ---------------------------------------------------------------------------


def attach_country_cadr_from_groups(
    lf_df: pd.DataFrame, group_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach country-level CADR totals and effective scaled ECA from group rows."""
    df = lf_df.copy()
    if group_df.empty:
        for col in _CADR_BACKFILL_COLUMNS:
            df[col] = np.nan
        return df

    agg = group_df.groupby("Country Code", as_index=False).agg(
        {
            INDOOR_ESSENTIAL_CADR_COL: "sum",
            INDOOR_VITAL_CADR_COL: "sum",
        }
    )
    df = df.drop(columns=list(_CADR_BACKFILL_COLUMNS), errors="ignore")
    df = df.merge(agg, on="Country Code", how="left")

    essential_workers = df["Indoor Essential Workers"]
    vital_workers = df["Indoor Vital Workers"]
    df[SCALED_ECA_ESSENTIAL_COL] = np.where(
        essential_workers > 0,
        df[INDOOR_ESSENTIAL_CADR_COL] / essential_workers,
        np.nan,
    )
    df[SCALED_ECA_VITAL_COL] = np.where(
        vital_workers > 0,
        df[INDOOR_VITAL_CADR_COL] / vital_workers,
        np.nan,
    )
    return df


def compute_global_worker_summary(
    lf_df: pd.DataFrame, lf_col: str = "Labour Force (2024)"
) -> pd.DataFrame:
    """Summarise global worker counts and shares of the labour force.

    Returns one row each for total Essential and Vital workers, plus indoor
    and outdoor splits for each. Outdoor counts are the residual of total
    minus indoor (jobs not classified as indoors under the ONET context
    projection).

    Parameters
    ----------
    lf_df:
        Country-level labour-force table after
        :func:`compute_absolute_counts` (must contain the four main count
        columns and ``lf_col``).
    lf_col:
        Column holding the labour-force denominator.

    Returns
    -------
    DataFrame
        Indexed by worker category with columns ``Workers`` (absolute count),
        ``% of Labour Force`` (global share), and the min/max country-level
        shares ``Country min %`` / ``Country max %``.
    """
    global_lf = lf_df[lf_col].sum(skipna=True)
    essential = lf_df["Essential Workers"].sum(skipna=True)
    vital = lf_df["Vital Workers"].sum(skipna=True)
    indoor_essential = lf_df["Indoor Essential Workers"].sum(skipna=True)
    indoor_vital = lf_df["Indoor Vital Workers"].sum(skipna=True)
    outdoor_essential = essential - indoor_essential
    outdoor_vital = vital - indoor_vital

    rows = {
        "Essential workers": essential,
        "Indoor essential workers": indoor_essential,
        "Outdoor essential workers": outdoor_essential,
        "Vital workers": vital,
        "Indoor vital workers": indoor_vital,
        "Outdoor vital workers": outdoor_vital,
    }

    # Country-level shares are stored as fractions; convert to percent for display.
    country_pct = {
        "Essential workers": lf_df["%Essential Workers"] * 100.0,
        "Indoor essential workers": lf_df["%Indoor Essential Workers"] * 100.0,
        "Outdoor essential workers": (
            lf_df["%Essential Workers"] - lf_df["%Indoor Essential Workers"]
        )
        * 100.0,
        "Vital workers": lf_df["%Vital Workers"] * 100.0,
        "Indoor vital workers": lf_df["%Indoor Vital Workers"] * 100.0,
        "Outdoor vital workers": (
            lf_df["%Vital Workers"] - lf_df["%Indoor Vital Workers"]
        )
        * 100.0,
    }

    summary = pd.DataFrame(
        {
            "Workers": rows,
            "% of Labour Force": {k: 100 * v / global_lf for k, v in rows.items()},
            "Country min %": {k: s.min(skipna=True) for k, s in country_pct.items()},
            "Country max %": {k: s.max(skipna=True) for k, s in country_pct.items()},
        }
    )
    summary.index.name = "Category"
    summary.attrs["labour_force"] = global_lf
    return summary


WORKER_PCT_RANK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("%Essential Workers", "% Essential"),
    ("%Indoor Essential Workers", "% Indoor essential"),
    ("%Vital Workers", "% Vital"),
    ("%Indoor Vital Workers", "% Indoor vital"),
)


def rank_countries_by_worker_pct(
    lf_df: pd.DataFrame,
    *,
    n: int = 10,
    columns: tuple[tuple[str, str], ...] = WORKER_PCT_RANK_COLUMNS,
    name_col: str = "Country Name",
    code_col: str = "Country Code",
) -> Dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Top and bottom ``n`` countries for each worker-share column.

    Returns a dict keyed by display label, each value a ``(top, bottom)``
    pair of DataFrames with columns ``Country Name``, ``Country Code``, and
    the share as a percent (0–100).
    """
    rankings: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for col, label in columns:
        ranked = (
            lf_df[[name_col, code_col, col]]
            .dropna(subset=[col])
            .assign(**{label: lf_df[col] * 100.0})
            .drop(columns=[col])
            .sort_values(label, ascending=False)
        )
        top = ranked.head(n).reset_index(drop=True)
        bottom = (
            ranked.tail(n).sort_values(label, ascending=True).reset_index(drop=True)
        )
        rankings[label] = (top, bottom)
    return rankings


def pitman_morgan_variance_test(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Pitman–Morgan test of equal variances for paired observations.

    Tests ``H0: Var(X) = Var(Y)`` via ``Corr(X+Y, X-Y) = 0`` (Pitman 1939;
    Morgan 1939), using :func:`scipy.stats.pearsonr`. Also returns one-sided
    ``p_y_smaller`` for ``Var(Y) < Var(X)``.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = int(xa.size)
    empty = {
        "n": float(n),
        "sd_x": float("nan"),
        "sd_y": float("nan"),
        "variance_ratio_y_over_x": float("nan"),
        "t_stat": float("nan"),
        "p_two_sided": float("nan"),
        "p_y_smaller": float("nan"),
    }
    if n < 3:
        return empty

    sd_x = float(np.std(xa, ddof=1))
    sd_y = float(np.std(ya, ddof=1))
    var_ratio = (sd_y * sd_y / (sd_x * sd_x)) if sd_x > 0 else float("nan")

    # H0: equal variances ⇔ Corr(X+Y, X-Y) = 0.
    r, p_two = stats.pearsonr(xa + ya, xa - ya)
    # One-sided H1: Var(Y) < Var(X) ⇔ r > 0.
    p_y_smaller = (p_two / 2.0) if r > 0 else (1.0 - p_two / 2.0)
    t_stat = r * np.sqrt((n - 2) / (1.0 - r * r)) if abs(r) < 1 else float("nan")

    return {
        "n": float(n),
        "sd_x": sd_x,
        "sd_y": sd_y,
        "variance_ratio_y_over_x": float(var_ratio),
        "t_stat": float(t_stat),
        "p_two_sided": float(p_two),
        "p_y_smaller": float(p_y_smaller),
    }


def paired_relative_spread_stats(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Relative-spread comparison for paired positive shares.

    Reports coefficients of variation (``CV = SD / mean``) and a
    Pitman–Morgan test on ``log(X)`` vs ``log(Y)``, which compares
    multiplicative / relative dispersion rather than absolute SD.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    xa, ya = xa[mask], ya[mask]
    n = int(xa.size)
    empty = {
        "n": float(n),
        "mean_x": float("nan"),
        "mean_y": float("nan"),
        "cv_x": float("nan"),
        "cv_y": float("nan"),
        "cv_ratio_y_over_x": float("nan"),
        "p_log_two_sided": float("nan"),
        "p_log_y_smaller": float("nan"),
    }
    if n < 3:
        return empty

    mean_x = float(np.mean(xa))
    mean_y = float(np.mean(ya))
    sd_x = float(np.std(xa, ddof=1))
    sd_y = float(np.std(ya, ddof=1))
    cv_x = sd_x / mean_x if mean_x > 0 else float("nan")
    cv_y = sd_y / mean_y if mean_y > 0 else float("nan")
    cv_ratio = cv_y / cv_x if cv_x and np.isfinite(cv_x) and cv_x > 0 else float("nan")

    log_pm = pitman_morgan_variance_test(np.log(xa), np.log(ya))
    return {
        "n": float(n),
        "mean_x": mean_x,
        "mean_y": mean_y,
        "cv_x": float(cv_x),
        "cv_y": float(cv_y),
        "cv_ratio_y_over_x": float(cv_ratio),
        "p_log_two_sided": log_pm["p_two_sided"],
        "p_log_y_smaller": log_pm["p_y_smaller"],
    }


def summarize_indoor_range_compression(
    lf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare country-level share ranges for total vs indoor worker categories.

    For each of Essential and Vital, reports the country min–max span (in
    percentage points) for the total share and the indoor share, then the
    absolute and relative change in that span when restricting to indoor
    workers. A negative ``% change in range`` means indoor shares are more
    similar across countries than total shares.

    Absolute spread: country-level SDs and Pitman–Morgan on the raw shares
    (``p (indoor SD < total)``).

    Relative spread: coefficients of variation (SD/mean) and Pitman–Morgan
    on log shares (``p (indoor log-SD < total)``), which asks whether
    multiplicative dispersion shrinks after means fall.
    """
    pairs = (
        (
            "Essential → Indoor essential",
            "%Essential Workers",
            "%Indoor Essential Workers",
        ),
        (
            "Vital → Indoor vital",
            "%Vital Workers",
            "%Indoor Vital Workers",
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, total_col, indoor_col in pairs:
        paired = lf_df[[total_col, indoor_col]].dropna()
        total_pct = paired[total_col] * 100.0
        indoor_pct = paired[indoor_col] * 100.0
        total_range = float(total_pct.max() - total_pct.min())
        indoor_range = float(indoor_pct.max() - indoor_pct.min())
        delta = indoor_range - total_range
        rel = (delta / total_range * 100.0) if total_range else float("nan")
        pm = pitman_morgan_variance_test(total_pct, indoor_pct)
        rel_spread = paired_relative_spread_stats(total_pct, indoor_pct)
        rows.append(
            {
                "Transition": label,
                "n countries": int(pm["n"]),
                "Total country min %": float(total_pct.min()),
                "Total country max %": float(total_pct.max()),
                "Total range (pp)": total_range,
                "Total SD (pp)": pm["sd_x"],
                "Total mean %": rel_spread["mean_x"],
                "Total CV": rel_spread["cv_x"],
                "Indoor country min %": float(indoor_pct.min()),
                "Indoor country max %": float(indoor_pct.max()),
                "Indoor range (pp)": indoor_range,
                "Indoor SD (pp)": pm["sd_y"],
                "Indoor mean %": rel_spread["mean_y"],
                "Indoor CV": rel_spread["cv_y"],
                "Δ range (pp)": delta,
                "% change in range": rel,
                "Variance ratio (indoor/total)": pm["variance_ratio_y_over_x"],
                "CV ratio (indoor/total)": rel_spread["cv_ratio_y_over_x"],
                "Pitman–Morgan p (two-sided)": pm["p_two_sided"],
                "p (indoor SD < total)": pm["p_y_smaller"],
                "p (indoor log-SD < total)": rel_spread["p_log_y_smaller"],
            }
        )
    return pd.DataFrame(rows).set_index("Transition")


# ---------------------------------------------------------------------------
# Food share of essential/vital workforce vs GDP per capita
# ---------------------------------------------------------------------------

GDP_PPP_COL = "GDP per capita, PPP (current international $)"
GDP_USD_COL = "GDP per capita (current US$)"
DEFAULT_GDP_PATH = ESSENTIAL_WORKERS_DATA / "GDP_per_capita_WDI.csv"

# Labour-force shares in EssentialWorkersByCountry (fractions 0–1).
LF_SHARE_COLS = (
    "%Essential Workers",
    "%Indoor Essential Workers",
    "%Vital Workers",
    "%Indoor Vital Workers",
)

# Food counts / category totals from EssentialWorkersByGroup.
FOOD_SHARE_COLS = (
    "Food % of Essential Workers",
    "Food % of Indoor Essential Workers",
    "Food % of Vital Workers",
    "Food % of Indoor Vital Workers",
)

_GROUP_COUNT_COLS = (
    "Essential Workers",
    "Vital Workers",
    "Indoor Essential Workers",
    "Indoor Vital Workers",
)

# Share of the essential / vital (indoor) workforce in each occupational group.
# Order matches ``_GROUP_COUNT_COLS``.
GROUP_COMPOSITION_SHARE_COLS = (
    "% of Essential Workers",
    "% of Vital Workers",
    "% of Indoor Essential Workers",
    "% of Indoor Vital Workers",
)


def _group_composition_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Attach within-scope composition shares to aggregated group counts.

    ``counts`` must include ``occupational_group`` and the four worker-count
    columns. Scope totals are the sum over groups in ``counts`` (caller must
    already restrict to one country, one region, or the world).
    """
    out = counts.copy()
    for count_col, share_col in zip(_GROUP_COUNT_COLS, GROUP_COMPOSITION_SHARE_COLS):
        total = float(out[count_col].sum())
        out[share_col] = out[count_col] / total if total > 0 else float("nan")
    # Stable group order matching GROUP_OVERLAP.
    order = {g: i for i, g in enumerate(GROUP_OVERLAP)}
    out["_ord"] = out["occupational_group"].map(order)
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


def summarize_group_composition(
    group_df: pd.DataFrame,
    *,
    by: str | None = None,
) -> pd.DataFrame:
    """Occupational-group breakdown of essential / vital workforces.

    Worker-weighted composition: each group's share is
    ``group_count / sum(groups)`` within the chosen scope.

    Parameters
    ----------
    group_df:
        Per-country × group counts (e.g. ``EssentialWorkersByGroup.csv``).
    by:
        ``None`` — one global composition (ILO-employment countries only).
        ``"Region"`` — composition within each UN region.
        ``"Country"`` — composition within each country (adds share columns
        to country × group rows).

    Returns
    -------
    DataFrame with count columns plus :data:`GROUP_COMPOSITION_SHARE_COLS`
    (fractions 0–1). For ``by="Country"`` / ``"Region"``, identifier columns
    are retained.
    """
    if group_df.empty:
        cols = ["occupational_group", *_GROUP_COUNT_COLS, *GROUP_COMPOSITION_SHARE_COLS]
        if by == "Region":
            cols = ["Region", *cols]
        elif by == "Country":
            cols = ["Country Name", "Country Code", "Region", *cols]
        return pd.DataFrame(columns=cols)

    if by is None:
        agg = group_df.groupby("occupational_group", as_index=False)[
            list(_GROUP_COUNT_COLS)
        ].sum()
        return _group_composition_from_counts(agg)

    if by == "Region":
        rows: list[pd.DataFrame] = []
        for region, gdf in group_df.groupby("Region", sort=True):
            agg = gdf.groupby("occupational_group", as_index=False)[
                list(_GROUP_COUNT_COLS)
            ].sum()
            part = _group_composition_from_counts(agg)
            part.insert(0, "Region", region)
            rows.append(part)
        return (
            pd.concat(rows, ignore_index=True)
            if rows
            else summarize_group_composition(group_df.iloc[0:0], by="Region")
        )

    if by == "Country":
        id_cols = [
            c
            for c in ("Country Name", "Country Code", "Region")
            if c in group_df.columns
        ]
        rows: list[pd.DataFrame] = []
        for keys, gdf in group_df.groupby(id_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            meta = dict(zip(id_cols, keys))
            part = _group_composition_from_counts(
                gdf[["occupational_group", *_GROUP_COUNT_COLS]].copy()
            )
            for col in reversed(id_cols):
                part.insert(0, col, meta[col])
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    raise ValueError("by must be None, 'Region', or 'Country'")


def food_share_of_workforce(group_df: pd.DataFrame) -> pd.DataFrame:
    """Per country, Food workers as a fraction of essential / vital totals.

    Denominators are the sum of all occupational groups in ``group_df`` for
    that country (matches country-level Essential/Vital counts where group
    data exist). Only countries present in ``group_df`` are returned.
    """
    id_cols = [
        c for c in ("Country Name", "Country Code", "Region") if c in group_df.columns
    ]
    if group_df.empty:
        return pd.DataFrame(columns=[*id_cols, *FOOD_SHARE_COLS])

    shares = summarize_group_composition(group_df, by="Country")
    food = shares.loc[shares["occupational_group"] == "Food"].copy()
    food = food.rename(
        columns={
            "% of Essential Workers": "Food % of Essential Workers",
            "% of Vital Workers": "Food % of Vital Workers",
            "% of Indoor Essential Workers": "Food % of Indoor Essential Workers",
            "% of Indoor Vital Workers": "Food % of Indoor Vital Workers",
        }
    )
    return food[[*id_cols, *FOOD_SHARE_COLS]].reset_index(drop=True)


def load_gdp_per_capita(
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Load World Bank WDI GDP per capita (latest year per country).

    Source file is built from indicators ``NY.GDP.PCAP.PP.CD`` (PPP) and
    ``NY.GDP.PCAP.CD`` (current USD). Prefer PPP for cross-country income
    comparisons.
    """
    gdp_path = Path(path) if path is not None else DEFAULT_GDP_PATH
    return pd.read_csv(gdp_path)


def correlate_share_with_gdp(
    df: pd.DataFrame,
    share_col: str,
    gdp_col: str = GDP_PPP_COL,
) -> dict[str, float]:
    """Spearman and Pearson(log GDP) association between a share and GDP."""
    sub = df[[share_col, gdp_col]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[(sub[gdp_col] > 0) & np.isfinite(sub[share_col])]
    n = int(len(sub))
    empty = {
        "n": float(n),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "pearson_log_gdp_r": float("nan"),
        "pearson_log_gdp_p": float("nan"),
    }
    if n < 3:
        return empty

    rho, rho_p = stats.spearmanr(sub[gdp_col], sub[share_col])
    log_gdp = np.log(sub[gdp_col].to_numpy(dtype=float))
    r, r_p = stats.pearsonr(log_gdp, sub[share_col].to_numpy(dtype=float))
    return {
        "n": float(n),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_log_gdp_r": float(r),
        "pearson_log_gdp_p": float(r_p),
    }


def summarize_worker_shares_vs_gdp(
    lf_df: pd.DataFrame,
    group_df: pd.DataFrame | None = None,
    gdp_df: pd.DataFrame | None = None,
    *,
    gdp_col: str = GDP_PPP_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join worker shares to GDP and return (merged table, correlation summary).

    Correlation summary covers:

    * labour-force shares: essential, indoor essential, vital, indoor vital
    * Food as a share of essential / vital (and indoor) workforce, when
      ``group_df`` is provided

    Primary association is Spearman (rank) vs GDP; Pearson on ``log(GDP)``
    is also reported. Negative rho supports larger shares in lower-income
    countries.
    """
    gdp = load_gdp_per_capita() if gdp_df is None else gdp_df.copy()
    keep = ["Country Code", gdp_col]
    for year_col in ("GDP Year (PPP)", "GDP Year (USD)"):
        if year_col in gdp.columns:
            keep.append(year_col)
    if GDP_USD_COL in gdp.columns and GDP_USD_COL != gdp_col:
        keep.append(GDP_USD_COL)
    gdp = gdp[[c for c in keep if c in gdp.columns]].drop_duplicates("Country Code")

    id_cols = [
        c for c in ("Country Name", "Country Code", "Region") if c in lf_df.columns
    ]
    share_cols = [c for c in LF_SHARE_COLS if c in lf_df.columns]
    merged = lf_df[id_cols + share_cols].merge(gdp, on="Country Code", how="left")

    food_cols: list[str] = []
    if group_df is not None and not group_df.empty:
        food = food_share_of_workforce(group_df)
        food_cols = [c for c in FOOD_SHARE_COLS if c in food.columns]
        merged = merged.merge(
            food[["Country Code"] + food_cols], on="Country Code", how="left"
        )

    rows: list[dict[str, Any]] = []
    for col in share_cols + food_cols:
        stats_row = correlate_share_with_gdp(merged, col, gdp_col=gdp_col)
        # Display labour-force shares and food shares as percent in the label.
        if col.startswith("%"):
            label = f"{col} (% of labour force)"
        else:
            label = col
        rows.append(
            {
                "Share": label,
                "column": col,
                "n": stats_row["n"],
                "Spearman ρ": stats_row["spearman_rho"],
                "Spearman p": stats_row["spearman_p"],
                "Pearson r (log GDP)": stats_row["pearson_log_gdp_r"],
                "Pearson p (log GDP)": stats_row["pearson_log_gdp_p"],
            }
        )
    summary = pd.DataFrame(rows)
    return merged, summary


# ---------------------------------------------------------------------------
# Final result: regional aggregation
# ---------------------------------------------------------------------------


_REGION_SUM_COLUMNS = [
    "Labour Force (2024)",
    "Indoor Essential Workers",
    "Indoor Vital Workers",
    "Essential Workers",
    "Vital Workers",
    "Armed Forces (Indoor Essential)",
    "Armed Forces (Essential)",
    INDOOR_ESSENTIAL_CADR_COL,
    INDOOR_VITAL_CADR_COL,
]


def aggregate_by_region(
    lf_df: pd.DataFrame, lf_col: str = "Labour Force (2024)"
) -> pd.DataFrame:
    """Sum per-country counts to UN regions and recompute the percentages."""
    regional = lf_df.groupby("Region")[_REGION_SUM_COLUMNS].sum().reset_index()
    for count_col, pct_col in _COUNT_COLUMNS:
        regional[pct_col] = regional[count_col] / regional[lf_col]
    regional[SCALED_ECA_ESSENTIAL_COL] = np.where(
        regional["Indoor Essential Workers"] > 0,
        regional[INDOOR_ESSENTIAL_CADR_COL] / regional["Indoor Essential Workers"],
        np.nan,
    )
    regional[SCALED_ECA_VITAL_COL] = np.where(
        regional["Indoor Vital Workers"] > 0,
        regional[INDOOR_VITAL_CADR_COL] / regional["Indoor Vital Workers"],
        np.nan,
    )
    return regional


# ---------------------------------------------------------------------------
# Validation against ILO published per-country %essential
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    merged_df: pd.DataFrame
    global_pct_essential: float
    global_pct_vital: float
    global_pct_indoor_essential: float
    global_pct_indoor_vital: float
    mean_our_pct_essential: float
    mean_ilo_pct_essential: float
    mean_abs_delta_pp: float
    correlation: float
    outlier_df: pd.DataFrame
    paper_average_pct: float = 51.72
    outlier_threshold_pp: float = 10.0


def validate_against_ilo(
    lf_df: pd.DataFrame,
    ilo_pct_df: pd.DataFrame,
    outlier_threshold_pp: float = 10.0,
    essential_pct_col: str = "%Essential Workers",
    our_pct_label: str = "Our %Essential (pct)",
) -> ValidationResult:
    """Compare our per-country %Essential to ILO's published figures.

    The "ground truth" here is the ILO WESO 2023 per-country "Share of
    key workers" column, which is computed from worker-level microdata
    using the full ISCO ∩ ISIC intersection. Our pipeline approximates
    that intersection with a single per-occupational-group overlap
    factor (see :data:`GROUP_OVERLAP` and the module docstring).
    Per-country deviations should therefore be read as the cost of
    that simplification, not as bugs in either calculation.

    The ``outlier_df`` field surfaces every country whose absolute
    deviation exceeds ``outlier_threshold_pp`` percentage points; the
    project's strict 10pp test (``tests/test_essential_workers.py::
    test_no_country_deviates_more_than_10pp``) currently fails on
    7 countries, the bulk of which sit in low-income agrarian
    economies where the "Manual" group's true ISIC overlap is far
    higher than the global 0.335 we use.
    """
    merged = lf_df.merge(ilo_pct_df, on="Country Name", how="inner")
    merged[our_pct_label] = merged[essential_pct_col] * 100
    merged["Delta (pp)"] = merged[our_pct_label] - merged["ILO %essential (published)"]

    global_lf = lf_df["Labour Force (2024)"].sum(skipna=True)
    global_essential = lf_df["Essential Workers"].sum(skipna=True)
    global_vital = lf_df["Vital Workers"].sum(skipna=True)
    global_indoor_essential = lf_df["Indoor Essential Workers"].sum(skipna=True)
    global_indoor_vital = lf_df["Indoor Vital Workers"].sum(skipna=True)

    outliers = merged.loc[merged["Delta (pp)"].abs() > outlier_threshold_pp].copy()
    outliers = outliers.reindex(
        outliers["Delta (pp)"].abs().sort_values(ascending=False).index
    )

    return ValidationResult(
        merged_df=merged,
        global_pct_essential=100 * global_essential / global_lf,
        global_pct_vital=100 * global_vital / global_lf,
        global_pct_indoor_essential=100 * global_indoor_essential / global_lf,
        global_pct_indoor_vital=100 * global_indoor_vital / global_lf,
        mean_our_pct_essential=float(merged[our_pct_label].mean()),
        mean_ilo_pct_essential=float(merged["ILO %essential (published)"].mean()),
        mean_abs_delta_pp=float(merged["Delta (pp)"].abs().mean()),
        correlation=float(
            merged[our_pct_label].corr(merged["ILO %essential (published)"])
        ),
        outlier_df=outliers,
        outlier_threshold_pp=outlier_threshold_pp,
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class EssentialWorkerOutputs:
    """All artefacts produced by :func:`run_pipeline`."""

    weights_df: pd.DataFrame
    employment_by_iso: Dict[str, Dict[str, float]]
    workers: WorkerDicts
    workers_model: WorkerDicts
    labour_force_df: pd.DataFrame
    group_df: pd.DataFrame
    regional_df: pd.DataFrame
    onsite_housing_df: pd.DataFrame
    ilo_pct_df: pd.DataFrame
    validation: ValidationResult
    validation_model: ValidationResult
    overlap_calibration: OverlapCalibrationResult


def compare_indoor_context_methods(
    data_dir: Path,
    methods: tuple[IndoorContextMethod, ...] = (
        "onet_max",
        "onet_banded",
        "jem_location",
    ),
) -> pd.DataFrame:
    """Run the pipeline under each indoor-context rule; return global summary rows."""
    rows = []
    for method in methods:
        out = run_pipeline(
            data_dir,
            write=False,
            indoor_context_method=method,
        )
        summary = compute_global_worker_summary(out.labour_force_df)
        row = summary.reset_index().rename(columns={"index": "Category"})
        row["indoor_context_method"] = method
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def run_pipeline(
    data_dir: Path,
    results_dir: Optional[Path] = None,
    write: bool = False,
    soc_to_isco_aggregator: str = "mean",
    indoor_context_method: IndoorContextMethod = "onet_banded",
    write_indoor_sensitivity: bool = False,
) -> EssentialWorkerOutputs:
    """Run the full essential-worker pipeline end-to-end.

    Parameters
    ----------
    data_dir:
        Directory containing the five source files described in the README
        (``ISCO-08 OpinionPollCensus.xlsx``,
        ``Indoors_Environmentally_Controlled_data.csv``,
        ``Indoors_Not_Environmentally_Controlled.csv``,
        ``ISCO_SOC_Crosswalk.csv``, ``ILO_ISCO_08_GLB.csv``,
        ``LFData_WB_plus.xlsx``, ``ILO_country_essential_workers_pct.xlsx``,
        and optionally ``job_exposure_matrix.xls`` for JEM-based indoor context).
    results_dir:
        Where to write CSV outputs when ``write`` is ``True``.
    write:
        If ``True``, write ``EssentialWorkersByCountry.csv``,
        ``EssentialWorkersByGroup.csv``, group-composition CSVs (global /
        region / country), ``EssentialWorkersByRegion.csv``,
        ``Essential_Workers_Validation.csv``,
        ``Group_Overlap_Calibration.csv``, and
        ``Onsite_Housing_Worker_Requirements.csv`` to ``results_dir``.

    Per-country group overlaps are calibrated to ILO published %essential
    (scalar ``x`` on global Figure A1 priors); vital workers use the same
    calibrated overlaps. Validation CSV includes model (global overlap) and
    calibrated series for the paper.
    soc_to_isco_aggregator:
        Passed through to :func:`build_isco_lvl2_weights`. Defaults to
        ``"mean"`` (corrected behaviour). Use ``"last"`` to reproduce the
        notebook's pre-refactor outputs exactly.
    indoor_context_method:
        ``onet_max`` (default): max of both O*NET indoor CSVs, linear %/100.
        ``onet_banded``: same source; 75–100% → 100% indoor, 50–75% → 50%, else 0.
        ``jem_location``: ``job_exposure_matrix.xls`` Location buckets.
    write_indoor_sensitivity:
        If ``True`` (with ``write``), also write ``Indoor_Context_Sensitivity.csv``.
    """
    data_dir = Path(data_dir)
    poll_df = pd.read_excel(
        data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl"
    )
    onet_controlled_df = pd.read_csv(
        data_dir / "Indoors_Environmentally_Controlled_data.csv"
    )
    onet_not_controlled_df = pd.read_csv(
        data_dir / "Indoors_Not_Environmentally_Controlled.csv"
    )
    crosswalk_df = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    ilo_df = pd.read_csv(data_dir / "ILO_ISCO_08_GLB.csv")
    lf_raw = pd.read_excel(data_dir / "LFData_WB_plus.xlsx", usecols=[0, 1, 3])
    jem_path = data_dir / "job_exposure_matrix.xls"

    weights_template = build_isco_lvl2_template(
        poll_df,
        crosswalk_df,
        onet_controlled_df=onet_controlled_df,
        onet_not_controlled_df=onet_not_controlled_df,
        indoor_context_method=indoor_context_method,
        jem_path=jem_path if jem_path.exists() else None,
        soc_to_isco_aggregator=soc_to_isco_aggregator,
    )
    weights = apply_group_overlaps(weights_template, GROUP_OVERLAP)
    employment_by_iso = build_employment_by_isco(ilo_df)
    ilo_pct_df = load_ilo_published_pct(
        data_dir / "ILO_country_essential_workers_pct.xlsx"
    )

    workers_model = compute_worker_dicts(employment_by_iso, weights_template)

    lf_df = prepare_labour_force(lf_raw)
    lf_df = fill_missing_labour_force_from_ilo_tot(lf_df, employment_by_iso)
    overlap_cal = calibrate_country_overlaps(
        lf_df, employment_by_iso, ilo_pct_df, weights_template
    )
    workers = compute_worker_dicts(
        employment_by_iso,
        weights_template,
        overlap_cal.overlaps_by_country,
    )
    overlap_cal.detail_df = build_group_overlap_calibration_detail(
        lf_df,
        overlap_cal.country_table,
        employment_by_iso,
        weights_template,
        ilo_pct_df,
        workers_model,
        workers,
    )

    lf_df = attach_pct_columns(lf_df, workers)
    lf_df = backfill_neighbours(lf_df)
    lf_df = attach_onsite_excluded_pct(
        lf_df,
        employment_by_iso,
        weights_template,
        overlap_cal.overlaps_by_country,
    )
    lf_df = backfill_neighbours(
        lf_df,
        cols=[ONSITE_EXCLUDED_ESSENTIAL_PCT_COL, ONSITE_EXCLUDED_VITAL_PCT_COL],
    )
    lf_df = compute_absolute_counts(lf_df)

    group_df = compute_group_workers_and_cadr(
        data_dir,
        lf_df,
        employment_by_iso,
        weights_template,
        overlap_cal.overlaps_by_country,
    )
    lf_df = attach_country_cadr_from_groups(lf_df, group_df)
    lf_df = backfill_neighbours(lf_df, cols=_CADR_BACKFILL_COLUMNS)

    regional_df = aggregate_by_region(lf_df)

    validation = validate_against_ilo(
        lf_df,
        ilo_pct_df,
        our_pct_label="Our %Essential (calibrated)",
    )
    lf_model = attach_pct_columns(lf_df.copy(), workers_model)
    validation_model = validate_against_ilo(
        lf_model,
        ilo_pct_df,
        our_pct_label="Our %Essential (model, global overlap)",
    )
    validation_merged = build_dual_validation_merged(
        lf_df, ilo_pct_df, workers_model, validation
    )
    onsite_housing_df = build_onsite_housing_worker_requirements(lf_df)

    if write:
        if results_dir is None:
            raise ValueError("results_dir is required when write=True")
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        lf_df.to_csv(results_dir / "EssentialWorkersByCountry.csv", index=False)
        group_df.to_csv(results_dir / "EssentialWorkersByGroup.csv", index=False)
        summarize_group_composition(group_df).to_csv(
            results_dir / "EssentialWorkersByGroupComposition_Global.csv", index=False
        )
        summarize_group_composition(group_df, by="Region").to_csv(
            results_dir / "EssentialWorkersByGroupComposition_ByRegion.csv",
            index=False,
        )
        summarize_group_composition(group_df, by="Country").to_csv(
            results_dir / "EssentialWorkersByGroupComposition_ByCountry.csv",
            index=False,
        )
        regional_df.to_csv(results_dir / "EssentialWorkersByRegion.csv", index=False)
        val_out_cols = [
            "Country Name",
            "Country Code",
            "Labour Force (2024)",
            "Essential Workers",
            "%Essential Workers",
            "Our %Essential (model, global overlap)",
            "Our %Essential (calibrated)",
            "ILO %essential (published)",
            "ILO %essential non-agri (published)",
            "Delta model (pp)",
            "Delta calibrated (pp)",
            "Armed Forces (Essential)",
        ]
        validation_merged[
            [c for c in val_out_cols if c in validation_merged.columns]
        ].to_csv(results_dir / "Essential_Workers_Validation.csv", index=False)
        overlap_cal.detail_df.to_csv(
            results_dir / "Group_Overlap_Calibration.csv", index=False
        )
        onsite_housing_df.to_csv(
            results_dir / "Onsite_Housing_Worker_Requirements.csv", index=False
        )
        if write_indoor_sensitivity:
            compare_indoor_context_methods(data_dir).to_csv(
                results_dir / "Indoor_Context_Sensitivity.csv", index=False
            )

    return EssentialWorkerOutputs(
        weights_df=weights,
        employment_by_iso=employment_by_iso,
        workers=workers,
        workers_model=workers_model,
        labour_force_df=lf_df,
        group_df=group_df,
        regional_df=regional_df,
        onsite_housing_df=onsite_housing_df,
        ilo_pct_df=ilo_pct_df,
        validation=validation,
        validation_model=validation_model,
        overlap_calibration=overlap_cal,
    )


if __name__ == "__main__":
    run_pipeline(
        ESSENTIAL_WORKERS_DATA,
        ESSENTIAL_WORKERS_RESULTS,
        write=True,
        write_indoor_sensitivity=True,
    )
