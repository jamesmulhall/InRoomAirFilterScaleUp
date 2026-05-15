"""CR Box / Coal-baghouse scale-up modelling.

This module re-implements the analysis previously embedded in
``scripts/Countries_Processing.ipynb`` as a small set of pure functions
plus a :func:`run_pipeline` orchestrator. It produces, for every country
and UN region, week-by-week CADR (Clean Air Delivery Rate) trajectories
from in-room filter manufacturing, repurposing, initial stock and coal
baghouse retrofits, plus Time-To-Reach (TTR) tables and the percentage of
indoor-vital workers covered after one year.

The notebook is now a thin walkthrough that imports from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from uncertainties import ufloat

from country_pkg import Country


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UN_REGION_LIST = [
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

# Per-filter CADR (L/s)
CR_Box_CADR_LS = 126.13
CADRPP = 100  # CADR per indoor-vital worker (L/s)

# Filter & manufacturing assumptions
Filter_Life_Span = ufloat((2 - 1) / 2, (2 - 1) / 4)
Scale_Up_Factor = 1 / 0.7
Initial_Stock_in_weeks = ufloat(6, 0.5)
Factory_minimum_production = 50000

# Coal-baghouse retrofit
Coalbaghouse_efficency = ufloat((0.8 + 0.5) / 2, (0.8 - 0.5) / 4)
Coalbaghouse_gradient = ufloat(1717, 419.3 / 2)
Coalbaghouse_offset = 33807
Coalbaghouse_Utilisation = ufloat(0.3, 0.1)

# Roll-out delays (weeks)
Repurposing_Delay = 12
Initial_Stock_Delay = 2
Coalbaghouse_Delay = 4
CR_Box_70_til_100_Delay = 6

# Distribution of repurposed CR Boxes across the active weeks.
REPUR_LIST = [0.04, 0.05, 0.06, 0.09, 0.13, 0.26, 0.13, 0.09, 0.06, 0.05, 0.04]

# Air-filter market reference values (USD/year, MERV breakdown).
Ind_Market_Rev_Per_MERV: Dict[str, float] = {
    "17-20": 2208.7e6,
    "5-8": 563.4e6,
    "9-12": 1271.8e6,
    "1-4": 171.7e6,
    "13-16": 1878.1e6,
}
Tot_Ind_Air_Filter = sum(Ind_Market_Rev_Per_MERV.values())
Tot_Air_Filter = 20.8303e9
All_Market_Rev_Per_MERV: Dict[str, float] = {
    k: v * Tot_Air_Filter / Tot_Ind_Air_Filter
    for k, v in Ind_Market_Rev_Per_MERV.items()
}
Price_Per_Filter: Dict[str, Any] = {
    "1-4": ufloat(1031.59, 107.26 / 2),
    "5-8": ufloat(1133.85, 447.48 / 2),
    "9-12": ufloat(1302.51, 554.53 / 2),
    "13-16": ufloat(1951.25, 593.63 / 2),
    "17-20": ufloat(22925.29, 3740.81 / 2),
}
Volume_to_Sale = 0.508 * 0.508 * 0.0254
Panel_Filter = ufloat(0.35, 0.35 * 0.1 / 2)


def _compute_sales(rev_per_merv: Dict[str, float]) -> Dict[str, Any]:
    """Convert per-MERV market revenue to physical filter counts."""
    return {
        k: rev_per_merv[k] / (Price_Per_Filter[k] * Volume_to_Sale)
        for k in rev_per_merv
    }


Sales_ALL = _compute_sales(All_Market_Rev_Per_MERV)
Sales_IND = _compute_sales(Ind_Market_Rev_Per_MERV)

# Derived filter availability used by the country calculations.
Usable_Filters = (
    (Sales_ALL["13-16"] + Sales_ALL["17-20"]) * Panel_Filter * Scale_Up_Factor
)
Repurposeable_Filters_ALL = (
    (Sales_ALL["13-16"] + Sales_ALL["17-20"]) * Panel_Filter * Filter_Life_Span
)
Repurposeable_Filters_IND = (
    (Sales_IND["13-16"] + Sales_IND["17-20"]) * Panel_Filter * Filter_Life_Span
)


# ---------------------------------------------------------------------------
# Country loading
# ---------------------------------------------------------------------------


def generate_countries_from_multiple_csvs(
    country_csv_path: Path,
    cr_box_csv_path: Optional[Path] = None,
    essential_workers_csv_path: Optional[Path] = None,
    baghouse_csv_path: Optional[Path] = None,
) -> Dict[str, Country]:
    """Build the ``{ISO-3: Country}`` dict from the four input CSVs."""
    df = pd.read_csv(country_csv_path, encoding="cp1252")
    for col in ("ISO-3", "Country Name"):
        if col not in df.columns:
            raise ValueError(f"CSV must have a column named '{col}'")

    cr_box_df = None
    if cr_box_csv_path:
        cr_box_df = pd.read_csv(cr_box_csv_path, encoding="cp1252")
        if "Country" not in cr_box_df.columns:
            raise ValueError("CR Box CSV must have a 'Country' column")
        cr_box_df["Country"] = cr_box_df["Country"].apply(
            Country._cc.convert, to="name_short"
        )

    essential_workers_df = None
    if essential_workers_csv_path:
        essential_workers_df = pd.read_csv(
            essential_workers_csv_path, encoding="cp1252"
        )
        for col in ("Country Name", "Country Code"):
            if col not in essential_workers_df.columns:
                raise ValueError(
                    f"Essential Workers CSV must have a column named '{col}'"
                )

    baghouse_df = None
    if baghouse_csv_path:
        baghouse_df = pd.read_csv(baghouse_csv_path, encoding="cp1252")
        for col in ("Country", "Operating MW"):
            if col not in baghouse_df.columns:
                raise ValueError(f"Baghouse CSV must have a column named '{col}'")
        baghouse_df["Country"] = baghouse_df["Country"].apply(
            Country._cc.convert, to="name_short"
        )

    countries: Dict[str, Country] = {}
    for _, row in df.iterrows():
        iso_code = row["ISO-3"]
        country_name = row["Country Name"]
        c = Country(name=iso_code)
        c.properties["ISO-3"] = iso_code

        if cr_box_df is not None:
            standardized_name = Country._cc.convert(country_name, to="name_short")
            cr_row = cr_box_df[cr_box_df["Country"] == standardized_name]
            if not cr_row.empty:
                for col in cr_row.columns:
                    if col != "Country":
                        c.properties[col] = cr_row.iloc[0][col]
            else:
                for col in cr_box_df.columns:
                    if col != "Country":
                        c.properties[col] = 0

        if essential_workers_df is not None:
            ew_row = essential_workers_df[
                essential_workers_df["Country Code"] == iso_code
            ]
            if not ew_row.empty:
                for col in ew_row.columns:
                    if col not in ("Country Code", "Country Name"):
                        c.properties[col] = ew_row.iloc[0][col]

        if baghouse_df is not None:
            standardized_name = Country._cc.convert(country_name, to="name_short")
            bh_row = baghouse_df[baghouse_df["Country"] == standardized_name]
            if not bh_row.empty:
                c.properties["Baghouse Operating MW"] = bh_row.iloc[0]["Operating MW"]
            else:
                c.properties["Baghouse Operating MW"] = 0

        countries[iso_code] = c
    return countries


# ---------------------------------------------------------------------------
# Manufacturing-delay step function
# ---------------------------------------------------------------------------


def manufacturing_delay_function(M: float) -> int:
    """Weeks of distribution delay as a step function of the MFS score."""
    if M > 90:
        return 1
    if M > 85:
        return 2
    if M > 80:
        return 3
    if M > 75:
        return 4
    if M > 70:
        return 5
    if M > 65:
        return 6
    if M > 60:
        return 7
    if M > 55.5:
        return 8
    return 0


# ---------------------------------------------------------------------------
# Per-country scale-up trajectories
# ---------------------------------------------------------------------------


def _cr_man_weekly_contribution(
    country: Country, week: int, cr_box_70_til_100_delay: int
) -> Any:
    """CR Box manufacturing contribution at a given week for one country."""
    mdd = country.properties["CR Box Manufacturing Distribution Delay"]
    if country.properties["Big_6"]:
        delay_til_100 = cr_box_70_til_100_delay + mdd
        if week < mdd:
            return 0
        if week < delay_til_100:
            return (0.7 + 0.05 * (week - 1)) * country.properties[
                "CADR: CR Box Weekly Production"
            ]
        return country.properties["CADR: CR Box Weekly Production"]
    if week >= mdd:
        return country.properties["CADR: CR Box Weekly Production"]
    return 0


def scale_up_CR_MAN(
    country: Country,
    weeks: int,
    cr_box_70_til_100_delay: int = CR_Box_70_til_100_Delay,
) -> List[Any]:
    """CR Box manufacturing scale-up trajectory (week 0 .. weeks)."""
    data = [0]
    for i in range(1, weeks + 1):
        data.append(
            data[-1] + _cr_man_weekly_contribution(country, i, cr_box_70_til_100_delay)
        )
    return data


def scale_up_CR_REPUR(
    country: Country,
    weeks: int,
    repur_list: List[float] = REPUR_LIST,
) -> List[Any]:
    """CR Box repurposing scale-up trajectory."""
    data = [0]
    repur_delay = country.properties["Repurposing Delay"]
    cadr_repur = country.properties["CADR: CR Box Repurposing"]
    for i in range(1, weeks + 1):
        contrib = 0
        if i < repur_delay:
            contrib = cadr_repur * repur_list[i - 1]
        data.append(data[-1] + contrib)
    return data


def scale_up_CR_STOCK(country: Country, weeks: int) -> List[Any]:
    """CR Box initial-stock release trajectory (single deposit)."""
    data = [0]
    stock_delay = country.properties["Initial Stock Delay"]
    cadr_stock = country.properties["CADR: CR Box Initial Stock"]
    for i in range(1, weeks + 1):
        contrib = cadr_stock if i == stock_delay else 0
        data.append(data[-1] + contrib)
    return data


def scale_up_COALBAG(country: Country, weeks: int) -> List[Any]:
    """Coal-baghouse retrofit trajectory (single deposit)."""
    data = [0]
    coal_delay = country.properties["Coalbaghouse Delay"]
    cadr_coal = country.properties["CADR: Coal Baghouse"]
    for i in range(1, weeks + 1):
        contrib = cadr_coal if i == coal_delay else 0
        data.append(data[-1] + contrib)
    return data


def scale_up_MAIN(
    country: Country,
    weeks: int,
    cr_box_70_til_100_delay: int = CR_Box_70_til_100_Delay,
    repur_list: List[float] = REPUR_LIST,
) -> List[Any]:
    """Combined CADR scale-up across all four supply streams."""
    cr_man = scale_up_CR_MAN(country, weeks, cr_box_70_til_100_delay)
    cr_repur = scale_up_CR_REPUR(country, weeks, repur_list)
    cr_stock = scale_up_CR_STOCK(country, weeks)
    coalbag = scale_up_COALBAG(country, weeks)
    return [a + b + c + d for a, b, c, d in zip(cr_man, cr_repur, cr_stock, coalbag)]


def compare_scale_up_data(
    data: List[Any], indoor_vital_count: float, cadrpp: float = CADRPP
) -> int:
    """First week at which ``data`` exceeds ``indoor_vital_count * cadrpp``.

    Matches the notebook semantics: when ``indoor_vital_count`` is 0 the
    threshold is never effectively crossed and the final week index is
    returned instead. Comparisons use the nominal value of any ufloat
    inputs (uncertainties' ``__gt__`` is being deprecated).
    """
    target = indoor_vital_count * cadrpp
    last_i = 0
    for i, value in enumerate(data, start=1):
        last_i = i
        nominal = getattr(value, "nominal_value", value)
        if indoor_vital_count != 0 and nominal > target:
            return i
    return last_i


# ---------------------------------------------------------------------------
# Country-level derived properties (CADR / Big_6 / delays)
# ---------------------------------------------------------------------------


REQUIRED_PROPERTIES = (
    "MSA",
    "MVA",
    "MFS",
    "%Indoor Essential Workers",
    "%Indoor Vital Workers",
    "Baghouse Operating MW",
)


def compute_country_properties(
    countries: Dict[str, Country],
    usable_filters: Any = Usable_Filters,
    repurposeable_filters_ind: Any = Repurposeable_Filters_IND,
    cr_box_cadr_ls: float = CR_Box_CADR_LS,
    initial_stock_in_weeks: Any = Initial_Stock_in_weeks,
    factory_minimum_production: float = Factory_minimum_production,
    coalbaghouse_efficency: Any = Coalbaghouse_efficency,
    coalbaghouse_utilisation: Any = Coalbaghouse_Utilisation,
    coalbaghouse_gradient: Any = Coalbaghouse_gradient,
    coalbaghouse_offset: float = Coalbaghouse_offset,
    repurposing_delay: int = Repurposing_Delay,
    initial_stock_delay: int = Initial_Stock_Delay,
    coalbaghouse_delay: int = Coalbaghouse_Delay,
) -> List[str]:
    """Compute every derived property used by the scale-up functions.

    Writes ``Big_6``, ``CR Box * Production``, ``CADR: ...`` and delay
    fields onto each ``Country`` in-place. Countries missing any of the
    inputs listed in :data:`REQUIRED_PROPERTIES` are silently skipped
    and removed from ``countries``; their ISO-3 codes are returned for
    the caller to log/inspect.
    """
    dropped: List[str] = []
    for iso, c in list(countries.items()):
        missing_any = any(prop not in c.properties for prop in REQUIRED_PROPERTIES)
        nan_any = any(
            pd.isna(c.properties.get(prop))
            for prop in REQUIRED_PROPERTIES
            if prop in c.properties
        )
        if missing_any or nan_any:
            dropped.append(iso)
            del countries[iso]

    sum_scale = 0
    for c in countries.values():
        if c.properties["MSA"] == 1:
            c.properties["Big_6"] = True
            sum_scale += c.properties["MVA"]
        else:
            c.properties["Big_6"] = False
    if sum_scale == 0:
        raise ValueError(
            "No Big_6 (MSA=1) countries found in the input - cannot scale."
        )
    scale = usable_filters / sum_scale

    global_mva = sum(c.properties["MVA"] for c in countries.values())

    for c in countries.values():
        msa = c.properties["MSA"]
        mva = c.properties["MVA"]
        x = scale * msa * mva if mva > 55.5 else ufloat(0, 0)
        x = ufloat(0, 0) if x.nominal_value < factory_minimum_production else x / 4
        c.properties["CR Box Annual Production"] = ufloat(
            math.floor(x.nominal_value), x.std_dev
        )
        c.properties["CR Box Weekly Production"] = (
            c.properties["CR Box Annual Production"] / 52
        )
        c.properties["CADR: CR Box Annual Production"] = (
            c.properties["CR Box Annual Production"] * cr_box_cadr_ls
        )
        c.properties["CADR: CR Box Weekly Production"] = (
            c.properties["CR Box Weekly Production"] * cr_box_cadr_ls
        )

        if c.properties["Big_6"]:
            c.properties["CR Box Initial Stock"] = (
                c.properties["CR Box Weekly Production"] * 0.7 * initial_stock_in_weeks
            )
            c.properties["CADR: CR Box Initial Stock"] = (
                c.properties["CR Box Initial Stock"] * cr_box_cadr_ls
            )
        else:
            c.properties["CR Box Initial Stock"] = ufloat(0, 0)
            c.properties["CADR: CR Box Initial Stock"] = ufloat(0, 0)

        c.properties["Ave Indoor Worker %"] = ufloat(
            (
                c.properties["%Indoor Essential Workers"]
                + c.properties["%Indoor Vital Workers"]
            )
            / 2,
            (
                c.properties["%Indoor Essential Workers"]
                - c.properties["%Indoor Vital Workers"]
            )
            / 4,
        )
        rel_mva = c.properties["MVA"] / global_mva
        c.properties["CR Box Repurposing"] = (
            (repurposeable_filters_ind / 4)
            * rel_mva
            * (1 - c.properties["Ave Indoor Worker %"])
        )
        c.properties["CADR: CR Box Repurposing"] = (
            c.properties["CR Box Repurposing"] * cr_box_cadr_ls
        )

        c.properties["CADR: Coal Baghouse"] = (
            coalbaghouse_efficency
            * coalbaghouse_utilisation
            * (
                coalbaghouse_gradient * c.properties["Baghouse Operating MW"]
                + coalbaghouse_offset
            )
        )

        c.properties["CR Box Manufacturing Distribution Delay"] = (
            manufacturing_delay_function(c.properties["MFS"])
        )
        c.properties["Repurposing Delay"] = repurposing_delay
        c.properties["Initial Stock Delay"] = initial_stock_delay
        c.properties["Coalbaghouse Delay"] = coalbaghouse_delay

    return dropped


# ---------------------------------------------------------------------------
# Per-country / per-region scale-up tables
# ---------------------------------------------------------------------------


@dataclass
class ScaleUpTables:
    """Country- and region-level scale-up trajectories.

    Each dictionary maps a country (or region) name to a list of values.
    The first two list entries are the indoor-vital and indoor-essential
    population counts; subsequent entries are the per-week CADR
    trajectory of length ``weeks + 1``.
    """

    weeks: int = 0
    country_main: Dict[str, list] = field(default_factory=dict)
    country_percent_indoor_vital: Dict[str, list] = field(default_factory=dict)
    country_cr_man: Dict[str, list] = field(default_factory=dict)
    country_cr_repur: Dict[str, list] = field(default_factory=dict)
    country_cr_stock: Dict[str, list] = field(default_factory=dict)
    country_coalbag: Dict[str, list] = field(default_factory=dict)
    region_main: Dict[str, list] = field(default_factory=dict)
    region_percent_indoor_vital: Dict[str, list] = field(default_factory=dict)
    region_cr_man: Dict[str, list] = field(default_factory=dict)
    region_cr_repur: Dict[str, list] = field(default_factory=dict)
    region_cr_stock: Dict[str, list] = field(default_factory=dict)
    region_coalbag: Dict[str, list] = field(default_factory=dict)


def _indoor_counts(country: Country) -> List[int]:
    return [
        (
            0
            if pd.isna(country.properties["Indoor Vital Workers"])
            else int(country.properties["Indoor Vital Workers"])
        ),
        (
            0
            if pd.isna(country.properties["Indoor Essential Workers"])
            else int(country.properties["Indoor Essential Workers"])
        ),
    ]


def _percent_indoor_vital(
    main_series: List[Any], indoor_vital_count: int, cadrpp: float = CADRPP
) -> List[Any]:
    if indoor_vital_count == 0:
        return [0] * len(main_series)
    return [100 * x / (cadrpp * indoor_vital_count) for x in main_series]


def scale_up_all_countries(
    countries: Dict[str, Country],
    weeks: int,
    un_regions: Tuple[str, ...] = tuple(UN_REGION_LIST),
    cadrpp: float = CADRPP,
) -> ScaleUpTables:
    """Build the per-country and per-region scale-up tables for ``weeks``."""
    out = ScaleUpTables(weeks=weeks)

    series_len = weeks + 3  # [vital_poll, essential_ilo, week0, week1, ..., weekN]
    for region in un_regions:
        out.region_main[region] = [0] * series_len
        out.region_cr_man[region] = [0] * series_len
        out.region_cr_repur[region] = [0] * series_len
        out.region_cr_stock[region] = [0] * series_len
        out.region_coalbag[region] = [0] * series_len

    for country in countries.values():
        main_pts = scale_up_MAIN(country, weeks)
        cr_man_pts = scale_up_CR_MAN(country, weeks)
        cr_repur_pts = scale_up_CR_REPUR(country, weeks)
        cr_stock_pts = scale_up_CR_STOCK(country, weeks)
        coalbag_pts = scale_up_COALBAG(country, weeks)
        indoor = _indoor_counts(country)

        out.country_main[country.name] = indoor + main_pts
        out.country_percent_indoor_vital[country.name] = indoor + _percent_indoor_vital(
            main_pts, indoor[0], cadrpp
        )
        out.country_cr_man[country.name] = indoor + cr_man_pts
        out.country_cr_repur[country.name] = indoor + cr_repur_pts
        out.country_cr_stock[country.name] = indoor + cr_stock_pts
        out.country_coalbag[country.name] = indoor + coalbag_pts

        region = country.properties.get("Region")
        if region in out.region_main:
            out.region_main[region] = list(
                np.add(out.country_main[country.name], out.region_main[region])
            )
            out.region_cr_man[region] = list(
                np.add(out.country_cr_man[country.name], out.region_cr_man[region])
            )
            out.region_cr_repur[region] = list(
                np.add(out.country_cr_repur[country.name], out.region_cr_repur[region])
            )
            out.region_cr_stock[region] = list(
                np.add(out.country_cr_stock[country.name], out.region_cr_stock[region])
            )
            out.region_coalbag[region] = list(
                np.add(out.country_coalbag[country.name], out.region_coalbag[region])
            )

    # Global = sum across all per-country series.
    global_main = [0] * series_len
    global_cr_man = [0] * series_len
    global_cr_repur = [0] * series_len
    global_cr_stock = [0] * series_len
    global_coalbag = [0] * series_len
    for name in out.country_main:
        global_main = list(np.add(global_main, out.country_main[name]))
        global_cr_man = list(np.add(global_cr_man, out.country_cr_man[name]))
        global_cr_repur = list(np.add(global_cr_repur, out.country_cr_repur[name]))
        global_cr_stock = list(np.add(global_cr_stock, out.country_cr_stock[name]))
        global_coalbag = list(np.add(global_coalbag, out.country_coalbag[name]))
    out.region_main["Global"] = global_main
    out.region_cr_man["Global"] = global_cr_man
    out.region_cr_repur["Global"] = global_cr_repur
    out.region_cr_stock["Global"] = global_cr_stock
    out.region_coalbag["Global"] = global_coalbag

    # % indoor vital per region (and globally) using the region's own
    # indoor-vital population total as the denominator.
    for region, series in out.region_main.items():
        indoor = series[:2]
        body = series[2:]
        out.region_percent_indoor_vital[region] = indoor + _percent_indoor_vital(
            body, indoor[0], cadrpp
        )
    return out


# ---------------------------------------------------------------------------
# Time-To-Reach
# ---------------------------------------------------------------------------


def time_to_reach(
    countries: Dict[str, Country],
    weeks: int = 52 * 5,
    un_regions: Tuple[str, ...] = tuple(UN_REGION_LIST),
    cadrpp: float = CADRPP,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Compute weeks-to-reach indoor-vital and indoor-essential thresholds.

    Returns ``(per_country, per_region)``.
    """
    region_indoor_vital: Dict[str, List[int]] = {}
    region_main: Dict[str, list] = {}
    per_country: Dict[str, List[int]] = {}

    for country in countries.values():
        data = scale_up_MAIN(country, weeks)
        indoor_vital = (
            0
            if pd.isna(country.properties["Indoor Vital Workers"])
            else int(country.properties["Indoor Vital Workers"])
        )
        indoor_essential = (
            0
            if pd.isna(country.properties["Indoor Essential Workers"])
            else int(country.properties["Indoor Essential Workers"])
        )
        region = country.properties.get("Region")
        if region in un_regions:
            region_indoor_vital[region] = list(
                np.add(
                    region_indoor_vital.get(region, [0, 0]),
                    [indoor_vital, indoor_essential],
                )
            )
            region_main[region] = list(
                np.add(region_main.get(region, [0] * (weeks + 1)), data)
            )
        per_country[country.name] = [
            compare_scale_up_data(data, indoor_vital, cadrpp),
            compare_scale_up_data(data, indoor_essential, cadrpp),
        ]

    per_region: Dict[str, List[int]] = {}
    for region, vals in region_indoor_vital.items():
        per_region[region] = [
            compare_scale_up_data(region_main[region], vals[0], cadrpp),
            compare_scale_up_data(region_main[region], vals[1], cadrpp),
        ]
    return per_country, per_region


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class CountriesOutputs:
    countries: Dict[str, Country]
    tables: ScaleUpTables
    main_df: pd.DataFrame
    percent_indoor_vital_df: pd.DataFrame
    cr_man_df: pd.DataFrame
    cr_repur_df: pd.DataFrame
    cr_stock_df: pd.DataFrame
    coalbag_df: pd.DataFrame
    ttr_country_df: pd.DataFrame
    ttr_region_df: pd.DataFrame
    pct_1y_region_df: pd.DataFrame
    pct_1y_country_df: pd.DataFrame


def _tables_to_df(
    country_data: Dict[str, list], region_data: Dict[str, list]
) -> pd.DataFrame:
    return pd.DataFrame({**country_data, **region_data}).T


def run_pipeline(
    data_dir: Path,
    results_dir: Path,
    essential_workers_csv: Optional[Path] = None,
    weeks: int = 52,
    ttr_weeks: int = 52 * 5,
    write: bool = False,
) -> CountriesOutputs:
    """Run the full Countries scale-up pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``STANDARD_COUNTRY_LIST.csv``,
        ``CR_Box_Countries_MS.csv`` and ``BaghouseAirflow.csv``.
    results_dir:
        Directory containing (or in which to write)
        ``EssentialWorkersByCountry.csv``.
    essential_workers_csv:
        Optional path override for the essential workers CSV; defaults to
        ``results_dir / 'EssentialWorkersByCountry.csv'``.
    weeks:
        Scale-up horizon for the main tables (default 52).
    ttr_weeks:
        Horizon for the time-to-reach calculations (default 5 years).
    write:
        If True, write all CSV/PKL outputs back into ``results_dir``.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    ew_csv = essential_workers_csv or (results_dir / "EssentialWorkersByCountry.csv")

    countries = generate_countries_from_multiple_csvs(
        data_dir / "STANDARD_COUNTRY_LIST.csv",
        data_dir / "CR_Box_Countries_MS.csv",
        ew_csv,
        data_dir / "BaghouseAirflow.csv",
    )
    compute_country_properties(countries)

    tables = scale_up_all_countries(countries, weeks=weeks)

    main_df = _tables_to_df(tables.country_main, tables.region_main)
    pct_df = _tables_to_df(
        tables.country_percent_indoor_vital, tables.region_percent_indoor_vital
    )
    cr_man_df = _tables_to_df(tables.country_cr_man, tables.region_cr_man)
    cr_repur_df = _tables_to_df(tables.country_cr_repur, tables.region_cr_repur)
    cr_stock_df = _tables_to_df(tables.country_cr_stock, tables.region_cr_stock)
    coalbag_df = _tables_to_df(tables.country_coalbag, tables.region_coalbag)

    ttr_country, ttr_region = time_to_reach(countries, weeks=ttr_weeks)
    ttr_country_df = pd.DataFrame.from_dict(
        ttr_country,
        orient="index",
        columns=["Indoor Vital in Weeks", "Indoor Essential in Weeks"],
    )
    ttr_country_df.index.name = "Region"
    ttr_region_df = pd.DataFrame.from_dict(
        ttr_region,
        orient="index",
        columns=["Indoor Vital in Weeks", "Indoor Essential in Weeks"],
    )
    ttr_region_df.index.name = "Region"

    pct_1y_region: Dict[str, float] = {}
    for region in UN_REGION_LIST:
        val = tables.region_percent_indoor_vital[region][-1]
        pct_1y_region[region] = (
            val.nominal_value if hasattr(val, "nominal_value") else val
        )
    pct_1y_country: Dict[str, float] = {}
    for name, series in tables.country_percent_indoor_vital.items():
        p = series[-1]
        if hasattr(p, "nominal_value") and p != 0:
            p = p.nominal_value
        if p != 0 and p > 100:
            p = 100
        pct_1y_country[name] = p

    pct_1y_region_df = pd.DataFrame.from_dict(
        pct_1y_region, orient="index", columns=["Percentage after 1 Year"]
    )
    pct_1y_region_df.index.name = "Region"
    pct_1y_country_df = pd.DataFrame.from_dict(
        pct_1y_country,
        orient="index",
        columns=["Countries Percentage after 1 Year"],
    )
    pct_1y_country_df.index.name = "Region"

    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        for df, stem in (
            (main_df, "Scale_up_output_MS"),
            (pct_df, "Scale_up_PERCENT_INDOOR_VITAL_MS"),
            (cr_man_df, "Scale_up_CR_MAN_MS"),
            (cr_repur_df, "Scale_up_CR_REPUR_MS"),
            (cr_stock_df, "Scale_up_CR_STOCK"),
            (coalbag_df, "Scale_up_COALBAG_MS"),
        ):
            df.to_csv(results_dir / f"{stem}.csv")
            df.to_pickle(results_dir / f"{stem}.pkl")

        ttr_country_df.to_csv(results_dir / "TTR_Country_MS.csv")
        ttr_region_df.to_csv(results_dir / "TTR_Region_MS.csv")
        pct_1y_region_df.to_csv(results_dir / "Percentage_After_1_Yr_region_MS.csv")
        pct_1y_country_df.to_csv(
            results_dir / "Country_Percentage_After_1_Yr_region_MS.csv"
        )

    return CountriesOutputs(
        countries=countries,
        tables=tables,
        main_df=main_df,
        percent_indoor_vital_df=pct_df,
        cr_man_df=cr_man_df,
        cr_repur_df=cr_repur_df,
        cr_stock_df=cr_stock_df,
        coalbag_df=coalbag_df,
        ttr_country_df=ttr_country_df,
        ttr_region_df=ttr_region_df,
        pct_1y_region_df=pct_1y_region_df,
        pct_1y_country_df=pct_1y_country_df,
    )
