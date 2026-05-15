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

In effect we assume the ISCO × ISIC overlap structure is **identical
across all countries**, taking the global mean as a proxy. It is not.
For example, a much larger share of "Manual" workers in low-income
agrarian economies (Liberia, Madagascar) are in essential agriculture
than in high-income economies, but our model uses the same 0.335
overlap factor for both. This is the dominant source of the
per-country deviations flagged by :func:`validate_against_ilo` (see
``tests/test_essential_workers.py::test_no_country_deviates_more_than_10pp``
which currently flags 7 countries above the 10pp threshold).

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
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import country_converter as coco
except ImportError as exc:  # pragma: no cover - required runtime dep
    raise ImportError(
        "country_converter is required for the essential_workers module. "
        "Install it via `pip install country_converter` or use the conda env."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
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

# ISCO-08 level-2 codes the ILO classes as a key (essential) occupation in
# Table A2 of WESO 2023 ("The value of essential work"). The list is the
# union of the 8 ILO occupational groups plus the 3 Armed Forces codes
# (01/02/03). Any L2 code NOT in this list automatically receives an
# Essential Weight ILO of 0 (and therefore contributes zero to the
# essential-worker totals).
#
# NOTE: this is only the OCCUPATION axis of the ILO definition. The
# industry (ISIC) axis is applied separately below via ``GROUP_OVERLAP``;
# see the module docstring for the full methodology.
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

# Map of ISCO-08 level-2 codes to the 8 ILO occupational groups + Armed
# Forces (Figure A1 of WESO 2023). The grouping matters because the
# ISCO × ISIC overlap is reported at the group level rather than at the
# L2 code level - we apply ``GROUP_OVERLAP[group]`` per group below.
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

# ISCO-08 level-2 codes that the team's in-house poll classified as
# "vital" but that ILO Table A2 explicitly excludes from key occupations
# on teleworkability grounds (e.g. 13 = ICT professionals, 21 = Science
# and engineering, 33 = Business and administration associates, 35 =
# ICT technicians). We zero out their poll vital weight so the Vital
# series stays aligned with the ILO's non-teleworkable scope.
NON_ILO_POLL_CODES = ["13", "21", "33", "35"]

# Armed-forces ISCO-08 level-2 codes used for diagnostic sub-totals.
ARMED_FORCES_L2 = ("01", "02", "03")

# Manual indoor-context overrides for the level-4 codes which would otherwise
# be missing context data (commissioned officer roles).
OVERRIDES_INDOOR_L4 = ["0110", "0210", "0310"]


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


# Shared country converter instance (creating it is expensive).
_CC = coco.CountryConverter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_country_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Standardise the top-level country keys of a nested dict."""
    return {
        _CC.convert(names=k, to="name_short", not_found="not found"): v
        for k, v in d.items()
    }


# ---------------------------------------------------------------------------
# 1. ISCO-08 level-2 weights
# ---------------------------------------------------------------------------


def build_isco_lvl2_weights(
    poll_df: pd.DataFrame,
    onet_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    soc_to_isco_aggregator: str = "mean",
) -> pd.DataFrame:
    """Build the ``ISCO_LVL2_WEIGHTS`` table.

    Parameters
    ----------
    poll_df:
        Wide poll spreadsheet with columns ``"ISCO-08"`` and ``"Census"``.
        ``ISCO-08`` may be int or str.
    onet_df:
        ONET indoor-context data with columns ``"Code"`` (SOC code, possibly
        with a ``.``-separated suffix) and ``"Context"`` (0-100 percent).
    crosswalk_df:
        SOC <-> ISCO-08 crosswalk with columns ``"2010 SOC Code"`` and
        ``"ISCO-08 Code"``.
    soc_to_isco_aggregator:
        How to combine multiple SOC codes that crosswalk to the same ISCO
        code (~43% of the ISCO codes touched have >1 SOC contribution).
        ``"mean"`` (default) averages all SOC contexts mapping to an ISCO
        code, which is the methodologically correct choice. ``"last"``
        keeps only the last SOC code seen in iteration order, which
        reproduces the notebook's pre-refactor behaviour exactly (the
        notebook's gating check compared SOC keys against ISCO keys and so
        never triggered the accumulate branch). Use ``"last"`` only when
        you need to regress against the older notebook outputs.

    Returns
    -------
    DataFrame
        Indexed by 2-digit ISCO-08 code (zero-padded string) with columns:

        ``Vital Weight POLL``
            Per-ISCO-L2 share of jobs the in-house poll classified as
            vital (averaged over L4 codes within the L2). Codes in
            :data:`NON_ILO_POLL_CODES` are forced to 0.
        ``Context Proj``
            Per-ISCO-L2 indoor-context fraction (0..1), averaged from
            the ONET indoor-environment data after the SOC -> ISCO L4
            crosswalk + L4 -> L2 collapse. Missing entries are filled
            from the next-coarsest ISCO ancestor (L3 -> L2 -> L1).
        ``Essential Weight ILO``
            Binary 1/0 - whether the L2 code appears in
            :data:`ILO_LVL2_ESSENTIAL_GROUPS` (ILO Table A2, occupation
            axis only - NOT the full ISCO ∩ ISIC intersection).
        ``Group`` / ``Group Overlap``
            ILO Figure A1 group assignment + its per-group ISIC × ISCO
            overlap factor from :data:`GROUP_OVERLAP`. ``Group Overlap``
            is the key per-country simplification - see the module
            docstring for caveats.
        ``ISCO_08_PollWeights``     = Vital × Context Proj × Group Overlap
        ``ISCO_08_PollWeights_Total`` = Vital × Group Overlap
        ``ISCO_08_ILOWeights``      = Essential × Context Proj × Group Overlap
        ``ISCO_08_ILOWeights_Total``  = Essential × Group Overlap

        The ``_Total`` columns drop the indoor filter and are used for
        the totals (Vital, Essential); the non-``_Total`` columns include
        Context Proj and feed Indoor Vital / Indoor Essential.
    """
    if soc_to_isco_aggregator not in ("mean", "last"):
        raise ValueError(
            f"soc_to_isco_aggregator must be 'mean' or 'last', got "
            f"{soc_to_isco_aggregator!r}"
        )
    # 1.1 Poll: keep just ISCO-08 (zero-padded L4) and Census
    poll = poll_df[["ISCO-08", "Census"]].copy()
    poll["ISCO-08"] = poll["ISCO-08"].astype(str).str.zfill(4)
    poll = poll.set_index("ISCO-08")

    # 1.2 ONET: normalise SOC codes and convert percent to fraction
    onet = onet_df[["Context", "Code"]].copy()
    onet["Code"] = onet["Code"].astype(str).str.split(".").str[0]
    onet["Context"] = onet["Context"] / 100
    onet_dict = onet.set_index("Code")["Context"].to_dict()

    # 1.3 SOC -> ISCO-08 crosswalk
    cw = crosswalk_df[["2010 SOC Code", "ISCO-08 Code"]].copy()
    cw["ISCO-08 Code"] = cw["ISCO-08 Code"].astype(str)
    soc_to_isco = cw.set_index("2010 SOC Code")["ISCO-08 Code"].to_dict()

    # Map ONET context onto ISCO-08 (L4) using the SOC-ISCO crosswalk. 43%
    # of touched ISCO codes are fed by multiple SOC codes; how to aggregate
    # them is controlled by ``soc_to_isco_aggregator``.
    isco_context_pool: Dict[str, list] = {}
    for k, v in onet_dict.items():
        if k in soc_to_isco:
            isco_context_pool.setdefault(soc_to_isco[k], []).append(v)

    if soc_to_isco_aggregator == "mean":
        isco_context_indoor = {
            k: float(np.mean(v)) for k, v in isco_context_pool.items()
        }
    else:  # "last"
        isco_context_indoor = {k: v[-1] for k, v in isco_context_pool.items()}

    isco_ctx_df = pd.DataFrame.from_dict(
        isco_context_indoor, orient="index", columns=["Context"]
    )
    isco_ctx_df.index.name = "ISCO-08 Code"

    # 1.4 Merge poll and ONET indoor context, then fill the missing entries
    # with the mean of the next-larger ISCO group (L3 -> L2 -> L1 -> global).
    left = poll.copy()
    right = isco_ctx_df.copy()
    left.index = left.index.astype(str).str.zfill(4)
    right.index = right.index.astype(str).str.zfill(4)

    df = pd.merge(left, right, left_index=True, right_index=True, how="left")
    df["Context"] = pd.to_numeric(df.get("Context"), errors="coerce")
    df["Census"] = pd.to_numeric(df.get("Census"), errors="coerce")

    df["_L3"] = df.index.str[:3]
    df["_L2"] = df.index.str[:2]
    df["_L1"] = df.index.str[:1]
    mean_l3 = df.groupby("_L3")["Context"].transform("mean")
    mean_l2 = df.groupby("_L2")["Context"].transform("mean")
    mean_l1 = df.groupby("_L1")["Context"].transform("mean")

    df["Context Proj"] = df["Context"].fillna(mean_l3).fillna(mean_l2).fillna(mean_l1)
    overrides_present = [c for c in OVERRIDES_INDOOR_L4 if c in df.index]
    df.loc[overrides_present, "Context Proj"] = 1
    df = df.drop(columns=["_L3", "_L2", "_L1", "Context"])

    # Collapse to level 2 by averaging both Census (poll) and Context Proj.
    lvl2 = (
        df.assign(_l2=df.index.str[:2])
        .groupby("_l2")
        .agg({"Census": "mean", "Context Proj": "mean"})
        .rename_axis("ISCO-08")
        .rename(columns={"Census": "Vital Weight POLL"})
    )

    # Essential weight from ILO Table A2 (binary 1/0).
    lvl2["Essential Weight ILO"] = (
        lvl2.index.astype(int).isin(ILO_LVL2_ESSENTIAL_GROUPS).astype(int)
    )

    # Farming carry-over: copy 62 onto 61 (Indoor/Outdoor context is missing
    # for code 61), then force subsistence farmers (63) to 100% outdoor.
    if "62" in lvl2.index:
        lvl2.loc["61"] = lvl2.loc["62"]
    if "63" in lvl2.index:
        lvl2.at["63", "Context Proj"] = 0

    # Suppress poll-vital codes the ILO methodology marks as teleworkable.
    for code in NON_ILO_POLL_CODES:
        if code in lvl2.index:
            lvl2.at[code, "Vital Weight POLL"] = 0

    # Attach group + group overlap then compute the four weight columns.
    # The Group Overlap multiplication below is how we approximate the
    # ISCO ∩ ISIC intersection in the ILO methodology. Because we apply
    # a single global factor per group to every country, this is the
    # main reason our per-country %Essential can diverge from ILO's
    # published per-country figures (see module docstring and
    # ``validate_against_ilo``).
    lvl2["Group"] = lvl2.index.map(ISCO_L2_TO_GROUP)
    lvl2["Group Overlap"] = lvl2["Group"].map(GROUP_OVERLAP).fillna(0.0)

    lvl2["ISCO_08_PollWeights"] = (
        lvl2["Vital Weight POLL"] * lvl2["Context Proj"] * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_ILOWeights"] = (
        lvl2["Essential Weight ILO"] * lvl2["Context Proj"] * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_PollWeights_Total"] = (
        lvl2["Vital Weight POLL"] * lvl2["Group Overlap"]
    )
    lvl2["ISCO_08_ILOWeights_Total"] = (
        lvl2["Essential Weight ILO"] * lvl2["Group Overlap"]
    )
    return lvl2


# ---------------------------------------------------------------------------
# 2. Employment by ISCO-08
# ---------------------------------------------------------------------------


def build_employment_by_isco(ilo_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Reduce the wide ILO ISCO-08 L2 employment CSV to ``{country: {code: employment}}``.

    For each (country, ISCO code) pair the latest year with a non-NaN value
    is kept. Country names are normalised via ``country_converter``.
    """
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

    nested = _normalise_country_keys(nested)

    for country, code_dict in nested.items():
        for code, year_dict in list(code_dict.items()):
            if isinstance(year_dict, dict) and year_dict:
                valid = {y: v for y, v in year_dict.items() if pd.notna(v)}
                code_dict[code] = valid[max(valid)] if valid else None

    return nested  # type: ignore[return-value]


@dataclass
class WorkerDicts:
    """Bundle of per-country worker counts and percentages."""

    iew_ilo: Dict[str, float] = field(default_factory=dict)
    ew_ilo: Dict[str, float] = field(default_factory=dict)
    ivw_poll: Dict[str, float] = field(default_factory=dict)
    vw_poll: Dict[str, float] = field(default_factory=dict)
    af_indoor_essential: Dict[str, float] = field(default_factory=dict)
    af_essential: Dict[str, float] = field(default_factory=dict)
    iew_pc: Dict[str, float] = field(default_factory=dict)
    ew_pc: Dict[str, float] = field(default_factory=dict)
    ivw_pc: Dict[str, float] = field(default_factory=dict)
    vw_pc: Dict[str, float] = field(default_factory=dict)
    af_indoor_essential_pc: Dict[str, float] = field(default_factory=dict)
    af_essential_pc: Dict[str, float] = field(default_factory=dict)


def compute_worker_dicts(
    employment_by_iso: Dict[str, Dict[str, float]],
    weights: pd.DataFrame,
) -> WorkerDicts:
    """Compute per-country indoor / total essential & vital worker counts."""
    poll_w = weights["ISCO_08_PollWeights"].to_dict()
    ilo_w = weights["ISCO_08_ILOWeights"].to_dict()
    poll_w_total = weights["ISCO_08_PollWeights_Total"].to_dict()
    ilo_w_total = weights["ISCO_08_ILOWeights_Total"].to_dict()

    out = WorkerDicts()

    for country, code_dict in employment_by_iso.items():
        iew = ew = ivw = vw = 0.0
        af_ind_e = af_e = 0.0
        for code, employment in code_dict.items():
            code_str = str(code).strip()
            if not pd.notna(employment):
                continue
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


def prepare_labour_force(lf_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the WB labour-force dataframe and attach UN regions."""
    df = lf_df.copy()
    df["Country Name"] = _CC.convert(
        df["Country Name"], to="short_name", not_found="not found"
    )
    df["Region"] = _CC.convert(df["Country Name"], to="UNregion", not_found="not found")
    return df


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


def compute_absolute_counts(
    lf_df: pd.DataFrame, lf_col: str = "Labour Force (2024)"
) -> pd.DataFrame:
    """Multiply each percentage column by the labour force to get counts."""
    df = lf_df.copy()
    for count_col, pct_col in _COUNT_COLUMNS:
        df[count_col] = df[pct_col] * df[lf_col]
    return df


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
]


def aggregate_by_region(
    lf_df: pd.DataFrame, lf_col: str = "Labour Force (2024)"
) -> pd.DataFrame:
    """Sum per-country counts to UN regions and recompute the percentages."""
    regional = lf_df.groupby("Region")[_REGION_SUM_COLUMNS].sum().reset_index()
    for count_col, pct_col in _COUNT_COLUMNS:
        regional[pct_col] = regional[count_col] / regional[lf_col]
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


def validate_against_ilo(
    lf_df: pd.DataFrame,
    ilo_pct_df: pd.DataFrame,
    outlier_threshold_pp: float = 10.0,
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
    merged["Our %Essential (pct)"] = merged["%Essential Workers"] * 100
    merged["Delta (pp)"] = (
        merged["Our %Essential (pct)"] - merged["ILO %essential (published)"]
    )

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
        mean_our_pct_essential=float(merged["Our %Essential (pct)"].mean()),
        mean_ilo_pct_essential=float(merged["ILO %essential (published)"].mean()),
        mean_abs_delta_pp=float(merged["Delta (pp)"].abs().mean()),
        correlation=float(
            merged["Our %Essential (pct)"].corr(merged["ILO %essential (published)"])
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
    labour_force_df: pd.DataFrame
    regional_df: pd.DataFrame
    ilo_pct_df: pd.DataFrame
    validation: ValidationResult


def run_pipeline(
    data_dir: Path,
    results_dir: Optional[Path] = None,
    write: bool = False,
    soc_to_isco_aggregator: str = "mean",
) -> EssentialWorkerOutputs:
    """Run the full essential-worker pipeline end-to-end.

    Parameters
    ----------
    data_dir:
        Directory containing the five source files described in the README
        (``ISCO-08 OpinionPollCensus.xlsx``,
        ``Indoors_Environmentally_Controlled_data.csv``,
        ``ISCO_SOC_Crosswalk.csv``, ``ILO_ISCO_08_GLB.csv``,
        ``LFData_WB_plus.xlsx`` and ``ILO_country_essential_workers_pct.xlsx``).
    results_dir:
        Where to write CSV outputs when ``write`` is ``True``.
    write:
        If ``True``, write ``EssentialWorkersByCountry.csv``,
        ``EssentialWorkersByRegion.csv`` and ``Essential_Workers_Validation.csv``
        to ``results_dir``.
    soc_to_isco_aggregator:
        Passed through to :func:`build_isco_lvl2_weights`. Defaults to
        ``"mean"`` (corrected behaviour). Use ``"last"`` to reproduce the
        notebook's pre-refactor outputs exactly.
    """
    data_dir = Path(data_dir)
    poll_df = pd.read_excel(
        data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl"
    )
    onet_df = pd.read_csv(data_dir / "Indoors_Environmentally_Controlled_data.csv")
    crosswalk_df = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    ilo_df = pd.read_csv(data_dir / "ILO_ISCO_08_GLB.csv")
    lf_raw = pd.read_excel(data_dir / "LFData_WB_plus.xlsx", usecols=[0, 1, 3])

    weights = build_isco_lvl2_weights(
        poll_df, onet_df, crosswalk_df, soc_to_isco_aggregator=soc_to_isco_aggregator
    )
    employment_by_iso = build_employment_by_isco(ilo_df)
    workers = compute_worker_dicts(employment_by_iso, weights)

    lf_df = prepare_labour_force(lf_raw)
    lf_df = attach_pct_columns(lf_df, workers)
    lf_df = backfill_neighbours(lf_df)
    lf_df = compute_absolute_counts(lf_df)

    regional_df = aggregate_by_region(lf_df)

    ilo_pct_df = load_ilo_published_pct(
        data_dir / "ILO_country_essential_workers_pct.xlsx"
    )
    validation = validate_against_ilo(lf_df, ilo_pct_df)

    if write:
        if results_dir is None:
            raise ValueError("results_dir is required when write=True")
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        lf_df.to_csv(results_dir / "EssentialWorkersByCountry.csv", index=False)
        regional_df.to_csv(results_dir / "EssentialWorkersByRegion.csv", index=False)
        val_out_cols = [
            "Country Name",
            "Country Code",
            "Labour Force (2024)",
            "Essential Workers",
            "%Essential Workers",
            "Our %Essential (pct)",
            "ILO %essential (published)",
            "ILO %essential non-agri (published)",
            "Delta (pp)",
            "Armed Forces (Essential)",
        ]
        validation.merged_df[val_out_cols].to_csv(
            results_dir / "Essential_Workers_Validation.csv", index=False
        )

    return EssentialWorkerOutputs(
        weights_df=weights,
        employment_by_iso=employment_by_iso,
        workers=workers,
        labour_force_df=lf_df,
        regional_df=regional_df,
        ilo_pct_df=ilo_pct_df,
        validation=validation,
    )
