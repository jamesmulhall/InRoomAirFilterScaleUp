"""
Scale-up of in-room air filtration supply during a severe airborne pandemic.

Implements methods section 2.3: commercial portable air cleaners (PACs), DIY
Corsi-Rosenthal (CR) boxes, and DIY units repurposed from coal-plant baghouse
filters. Global annual production is estimated for each channel, allocated
across countries by manufacturing value added, converted to clean air delivery
rate (eCADR), and spread over a weekly timeline. Outputs are weekly eCADR by
country and by supply channel, and the weekly share of the essential and vital
workforces covered globally and by UN region.

All uncertain parameters are read from data/parameters.csv and all fixed
settings from data/settings.csv, so no numbers are hard-coded here. The two
fitted linear models (coal airflow, MVA exponent) come from linear_models.py.

All three scale-up scenarios are run and written out on every execution.
"""

import os

import numpy as np
import pandas as pd
import requests
import country_converter as coco

from essential_workers import backfill_neighbours
from mc_distributions import sample_normal, sample_lognormal, sample_uniform

# Input and output locations
PARAM_FILE = "data/scale_up/parameters.csv"
SETTINGS_FILE = "data/scale_up/settings.csv"
MVA_CACHE = "data/scale_up/mva_world_bank.csv"
MVA_INDICATOR = "NV.IND.MANF.CD"
MVA_URL = f"https://api.worldbank.org/v2/country/all/indicator/{MVA_INDICATOR}"
MVA_YEARS = "2010:2024"
BAGHOUSE_FILE = "data/scale_up/BaghouseAirflow.csv"
COMTRADE_FILE = "data/scale_up/comtrade_HS842139.xlsx"
WORKERS_FILE = "results/essential_workers/EssentialWorkersByCountry.csv"
RESULTS_DIR = "results/scale_up"

# Channels carried through the model
CHANNELS = [
    "pac",
    "cr_box",
    "baghouse",
    "repurposed_pac",
    "repurposed_cr_box",
    "repurposed_baghouse",
]

# MERV bands the industrial air filter market is broken down into, and the
# subset that meets the MERV 13 requirement for CR boxes
MERV_BANDS = ["1_4", "5_8", "9_12", "13_16", "17_20"]
MERV13_BANDS = ["13_16", "17_20"]

# Weeks summarised on the console; the coverage files cover every week
REPORT_WEEKS = [13, 26]

# Scenarios from methods 2.3.4
SCENARIOS = {
    1: "utilisation only",
    2: "meltblown capped",
    3: "COVID analogue growth",
}


def load_parameters(path=PARAM_FILE):
    """
    Read the uncertain parameter table.

    Arguments:
        path (str): Path to the parameter CSV.

    Returns:
        pandas.DataFrame: Indexed by parameter, with low, high and distribution.
    """
    params = pd.read_csv(path).set_index("parameter")
    missing = params[(params.low == 0) & (params.high == 0)].index.tolist()
    if missing:
        raise ValueError(
            "These parameters have no value in the methods and must be set in "
            f"{path} before the model can run:\n  " + "\n  ".join(missing)
        )
    return params


def load_settings(path=SETTINGS_FILE):
    """
    Read the fixed settings table.

    Arguments:
        path (str): Path to the settings CSV.

    Returns:
        dict: Setting name to value, numeric where possible.
    """
    table = pd.read_csv(path).set_index("setting")["value"]
    settings = {}
    for name, value in table.items():
        lowered = str(value).strip().lower()
        if lowered in ("true", "false"):
            settings[name] = lowered == "true"
            continue
        try:
            settings[name] = float(value)
        except ValueError:
            settings[name] = value
    return settings


def sample_all(params, n):
    """
    Sample every parameter once, using each one's stated distribution.

    Sampling up front rather than inside each calculation means every scenario
    sees identical draws, so scenarios can be compared draw for draw.

    Arguments:
        params (pandas.DataFrame): Output of load_parameters.
        n (int): Number of draws.

    Returns:
        dict: Parameter name to samples, each shape (n,).
    """
    samples = {}
    for name, row in params.iterrows():
        low, high, dist = row["low"], row["high"], row["distribution"]
        if low == high:
            samples[name] = np.full(n, float(low))
        elif dist == "normal":
            samples[name] = sample_normal(low, high, n)
        elif dist == "lognormal":
            samples[name] = sample_lognormal(low, high, n)
        elif dist == "uniform":
            samples[name] = sample_uniform(low, high, n)
        else:
            raise ValueError(f"Unknown distribution '{dist}' for parameter '{name}'")
    return samples


def fetch_mva(cache=MVA_CACHE, years=MVA_YEARS, refresh=False):
    """
    Manufacturing value added per country, from the World Bank API.

    Takes the most recent year with a positive value for each country, so
    coverage is not limited to the latest reporting year. The result is cached
    to disk, so the model runs offline after the first call. World Bank
    aggregates such as "World" are left in the cache and drop out when the
    table is merged onto the model's country list.

    Arguments:
        cache (str): Path to the cached CSV.
        years (str): Year range to request, as "first:last".
        refresh (bool): Re-download even if the cache exists.

    Returns:
        pandas.DataFrame: Columns iso3, mva_usd and mva_year.
    """
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache)

    print(f"  downloading {MVA_INDICATOR} from the World Bank ({years})...")
    records, page = [], 1
    while True:
        response = requests.get(
            MVA_URL,
            params={"format": "json", "date": years, "per_page": 20000, "page": page},
            timeout=120,
        )
        response.raise_for_status()
        header, batch = response.json()
        records.extend(batch or [])
        if page >= header["pages"]:
            break
        page += 1

    df = pd.DataFrame(
        [
            {
                "iso3": r["countryiso3code"],
                "mva_year": int(r["date"]),
                "mva_usd": r["value"],
            }
            for r in records
            if r["value"] is not None and r["value"] > 0 and r["countryiso3code"]
        ]
    )
    df = df.sort_values("mva_year").groupby("iso3", as_index=False).last()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  cached {len(df)} countries to {cache}")
    return df


def load_filter_export_value_per_kg(path=COMTRADE_FILE):
    """
    USD per kilogram of HS 842139 gas-filter exports, from UN Comtrade.

    Value per kilogram is ``primaryValue / netWgt``. Rows without a positive
    weight or value are dropped. If more than one year is present, each
    reporter keeps its latest year.

    Arguments:
        path (str): Path to the Comtrade Excel extract.

    Returns:
        pandas.DataFrame: Columns iso3 and value_per_kg.
    """
    trade = pd.read_excel(path)
    usable = trade[(trade.netWgt > 0) & (trade.primaryValue > 0)].copy()
    usable["value_per_kg"] = usable.primaryValue / usable.netWgt
    latest = usable.sort_values("refYear").groupby("reporterISO", as_index=False).last()
    return latest.rename(columns={"reporterISO": "iso3"})[["iso3", "value_per_kg"]]


def divide_mva_by_export_price(df, value_per_kg, similar_iso3=None):
    """
    Divide manufacturing value added by export value per kilogram.

    Missing prices are filled the same way as other country gaps in this
    repository: the mean of ``SIMILAR_ISO3`` neighbours, then the median of
    countries that still have a price (the overlap pipeline's global
    fallback). Countries with no MVA stay at zero.

    Arguments:
        df (pandas.DataFrame): Country table with iso3 and mva_usd.
        value_per_kg (pandas.DataFrame): Output of load_filter_export_value_per_kg.
        similar_iso3 (dict): Neighbour map; the pipeline default is used if
            omitted.

    Returns:
        pandas.DataFrame: The country table with mva_usd in quantity units.
    """
    out = df.merge(value_per_kg, on="iso3", how="left")
    from_trade = out.value_per_kg.notna()
    filled = backfill_neighbours(
        out.rename(columns={"iso3": "Country Code"}),
        similar_iso3=similar_iso3,
        cols=["value_per_kg"],
    ).rename(columns={"Country Code": "iso3"})
    from_neighbour = filled.value_per_kg.notna() & ~from_trade
    fallback = filled.value_per_kg.median()
    from_global = filled.value_per_kg.isna()
    filled["value_per_kg"] = filled["value_per_kg"].fillna(fallback)
    print(
        f"  filter export USD/kg: {from_trade.sum()} from Comtrade, "
        f"{from_neighbour.sum()} from neighbours, "
        f"{from_global.sum()} from the global median ({fallback:.1f})"
    )
    has_mva = filled.mva_usd > 0
    filled.loc[has_mva, "mva_usd"] = (
        filled.loc[has_mva, "mva_usd"] / filled.loc[has_mva, "value_per_kg"]
    )
    return filled.drop(columns=["value_per_kg"])


def load_country_data(adjust_mva_by_cost=False):
    """
    Build the country table from manufacturing, coal and workforce inputs.

    Arguments:
        adjust_mva_by_cost (bool): If True, divide MVA by Comtrade HS 842139
            export value per kilogram so cheaper producers get a larger
            allocation.

    Returns:
        pandas.DataFrame: One row per country with manufacturing value added,
            coal capacity, workforce shares and the eCADR requirements.
    """
    print("Loading country data...")
    converter = coco.CountryConverter()

    workers = pd.read_csv(WORKERS_FILE, encoding="cp1252")
    workers = workers.rename(columns={"Country Code": "iso3", "Region": "region"})

    mva = fetch_mva()

    baghouse = pd.read_csv(BAGHOUSE_FILE, encoding="cp1252")
    baghouse["iso3"] = converter.convert(
        baghouse.Country.tolist(), to="ISO3", not_found=None
    )

    df = workers[
        [
            "iso3",
            "Country Name",
            "region",
            "%Essential Workers",
            "Indoor Essential CADR Requirement (L/s)",
            "Indoor Vital CADR Requirement (L/s)",
        ]
    ].copy()
    df = df.merge(mva[["iso3", "mva_usd", "mva_year"]], on="iso3", how="left")
    df = df.merge(baghouse[["iso3", "Operating MW"]], on="iso3", how="left")
    df[["mva_usd", "Operating MW"]] = df[["mva_usd", "Operating MW"]].fillna(0)
    df = df.dropna(subset=["region"]).reset_index(drop=True)

    if adjust_mva_by_cost:
        print("  dividing MVA by HS 842139 export value per kilogram")
        df = divide_mva_by_export_price(df, load_filter_export_value_per_kg())

    no_mva = (df.mva_usd <= 0).sum()
    print(
        f"  {len(df)} countries; {no_mva} have no reported MVA and cannot manufacture"
    )
    cutoff = int(df.mva_year.max()) - 2
    stale = df[(df.mva_usd > 0) & (df.mva_year < cutoff)]
    if not stale.empty:
        print(f"  {len(stale)} countries rely on MVA older than {cutoff}")
    return df


def production_shares(df, exponent, min_production, total_filters):
    """
    Country share of global production, proportional to MVA raised to a power.

    Implements methods equation 3. Countries whose implied total filter output
    falls below the minimum are set to zero and their share is redistributed
    across the countries above it.

    Arguments:
        df (pandas.DataFrame): Country table with an mva_usd column.
        exponent (float): The exponent b.
        min_production (float): Minimum national total filter production.
        total_filters (float): Global total filter production, used to test the
            threshold.

    Returns:
        numpy.ndarray: Shares summing to one, shape (n_countries,).
    """
    has_mva = df.mva_usd.to_numpy(float) > 0
    weights = np.where(has_mva, df.mva_usd.to_numpy(float) ** exponent, 0.0)
    shares = weights / weights.sum()

    below = has_mva & (shares * total_filters < min_production)
    shares = np.where(below, 0.0, shares)
    print(f"  {has_mva.sum()} countries with MVA, {(~has_mva).sum()} without")
    print(
        f"  {below.sum()} of {has_mva.sum()} fall below the {min_production:,.0f} "
        f"unit threshold ({1 - shares.sum():.2%} of production reallocated)"
    )
    return shares / shares.sum()


def pac_panel_filter_units(samples):
    """
    Panel-format PAC units per year.

    These are the units that go to CR boxes when ``prioritize_cr_boxes`` is
    True, and stay with PACs when it is False.

    Arguments:
        samples (dict): Output of sample_all.

    Returns:
        numpy.ndarray: PAC units per year, shape (n,).
    """
    return (
        samples["pac_market_revenue_usd"]
        * samples["fraction_room"]
        * samples["fraction_merv13_plus"]
        * (1 - samples["fraction_non_panel"])
        / samples["pac_price_usd"]
    )


def pac_panel_cr_box_filters(samples):
    """
    Panel-format PAC units expressed as 20x20x1 filter counts.

    PAC panel filters are smaller than the filters CR boxes use (equation 4).
    Each PAC unit is scaled by the ratio of filter volumes before subtracting
    from the CR box filter pool.

    Arguments:
        samples (dict): Output of sample_all.

    Returns:
        numpy.ndarray: Equivalent 20x20x1 filters per year, shape (n,).
    """
    volume_ratio = samples["pac_panel_filter_volume_m3"] / samples["filter_volume_m3"]
    return pac_panel_filter_units(samples) * volume_ratio


def pac_ecadr_global(samples, prioritize_cr_boxes=False):
    """
    Global annual eCADR from commercial portable air cleaners.

    Implements methods equations 1 and 2. When ``prioritize_cr_boxes`` is
    True, panel-format units are left out of PAC production and counted
    toward CR boxes instead. When False (the default), PACs take the full
    eligible share and those panel units are subtracted from the CR box
    filter pool.

    Arguments:
        samples (dict): Output of sample_all.
        prioritize_cr_boxes (bool): If True, redirects panel filters to CR
            boxes instead of PACs. If False, panel filters stay with PACs.

    Returns:
        numpy.ndarray: eCADR in L/s per year of production, shape (n,).
    """
    units_non_panel = (
        samples["pac_market_revenue_usd"]
        * samples["fraction_room"]
        * samples["fraction_merv13_plus"]
        * samples["fraction_non_panel"]
        / samples["pac_price_usd"]
    )
    units_panel_pac = pac_panel_filter_units(samples)
    units_total = units_non_panel + units_panel_pac
    if prioritize_cr_boxes:
        return units_non_panel * samples["pac_ecadr_l_per_s"]
    return units_total * samples["pac_ecadr_l_per_s"]


def merv_revenue_shares(samples, bands=MERV13_BANDS):
    """
    Share of air filter revenue in each MERV band.

    Implements the first step of methods equation 4: the split of revenue by
    MERV rating is taken from industrial air filter market data, then applied
    to whichever revenue base the caller supplies. Shares are always fractions
    of the whole market, so asking for a subset of bands returns less than one.

    Arguments:
        samples (dict): Output of sample_all.
        bands (list): Bands to return, from MERV_BANDS.

    Returns:
        dict: Band name to revenue share, each shape (n,).
    """
    revenue = {b: samples[f"industrial_revenue_{b}_usd"] for b in MERV_BANDS}
    total = sum(revenue.values())
    return {b: revenue[b] / total for b in bands}


def filter_production(samples, market_revenue, bands=MERV13_BANDS):
    """
    Global annual production of panel filters from a revenue base.

    Implements methods equation 4. Revenue in each MERV band is divided by the
    price per cubic metre of media times the filter volume to give filter counts.
    Passing MERV_BANDS gives total panel filter production, which is what the
    minimum national threshold in production_shares is defined on.

    Arguments:
        samples (dict): Output of sample_all.
        market_revenue (numpy.ndarray): Air filter revenue to split, shape (n,).
        bands (list): MERV bands to count, from MERV_BANDS.

    Returns:
        numpy.ndarray: Filters per year, shape (n,).
    """
    panel = samples["fraction_panel"]
    volume = samples["filter_volume_m3"]
    filters = np.zeros_like(market_revenue)
    for band, share in merv_revenue_shares(samples, bands).items():
        price_per_volume = samples[f"price_per_volume_{band}"]
        filters += market_revenue * share * panel / (price_per_volume * volume)
    return filters


def industrial_revenue_total(samples):
    """
    Total industrial air filter market revenue, summed across MERV bands.

    Used for methods equation 10, where repurposing draws only on filters in
    industrial settings because residential panel filters are left in place.

    Arguments:
        samples (dict): Output of sample_all.

    Returns:
        numpy.ndarray: Revenue in USD per year, shape (n,).
    """
    return sum(samples[f"industrial_revenue_{b}_usd"] for b in MERV_BANDS)


def fan_production(samples, settings):
    """
    Global annual fan production, expressed as the CR boxes it could equip.

    Implements the fan term of methods equation 5. Both fan types are sized by
    dividing market revenue by the average price per fan. A box fan equips one
    CR box; PC fans are counted in the number needed per box.

    Arguments:
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.

    Returns:
        numpy.ndarray: CR boxes that could be equipped per year, shape (n,).
    """
    pc_fans = samples["pc_fan_market_revenue_usd"] / samples["pc_fan_price_usd"]
    box_fans = samples["box_fan_market_revenue_usd"] / samples["box_fan_price_usd"]
    return pc_fans / settings["pc_fans_per_cr_box"] + box_fans


def cr_box_ecadr_global(samples, settings, market_revenue, panel_pac_deduction=None):
    """
    Global annual eCADR from CR boxes, limited by filters or fans.

    Implements methods equations 5 and 6.

    Arguments:
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.
        market_revenue (numpy.ndarray): Air filter revenue base, shape (n,).
        panel_pac_deduction (numpy.ndarray): 20x20x1-equivalent filters to
            subtract from the pool when PACs keep their panel-format share.

    Returns:
        tuple: (eCADR in L/s per year, filters per year, limiting component).
    """
    filters = filter_production(samples, market_revenue)
    if panel_pac_deduction is not None:
        filters = np.maximum(filters - panel_pac_deduction, 0.0)
    from_filters = filters / settings["filters_per_cr_box"]
    from_fans = fan_production(samples, settings)
    boxes = np.minimum(from_filters, from_fans)
    limiting = "filters" if from_filters.mean() < from_fans.mean() else "fans"
    return boxes * samples["cr_box_ecadr_l_per_s"], filters, limiting


def baghouse_standing_ecadr(df, samples, settings):
    """
    Total eCADR held in the world's installed coal baghouse filters.

    Implements methods equations 7 and 8. The fitted intercept is per-plant, so
    it is only applied to countries that actually have coal capacity. The result
    is summed globally and reallocated by MVA alongside the other channels.

    Arguments:
        df (pandas.DataFrame): Country table with an Operating MW column.
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.
        n (int): Number of draws.

    Returns:
        numpy.ndarray: Global standing eCADR in L/s, shape (n,).
    """
    megawatts = df["Operating MW"].to_numpy(float)
    airflow = np.where(
        megawatts > 0,
        settings["baghouse_gradient"] * megawatts
        + settings["baghouse_intercept_l_per_s"],
        0.0,
    )
    return (
        airflow.sum() * samples["baghouse_efficiency"] * samples["baghouse_utilisation"]
    )


def repurposed_ecadr(annual_ecadr, lifespan, essential_share, recovered):
    """
    eCADR recovered from units already in service in non-essential workplaces.

    Implements methods equations 9 and 10.

    Arguments:
        annual_ecadr (numpy.ndarray): Annual production eCADR, shape (n, n_countries).
        lifespan (numpy.ndarray): Unit lifespan in years, shape (n,).
        essential_share (numpy.ndarray): Essential fraction, shape (n_countries,).
        recovered (numpy.ndarray): Fraction of units recovered, shape (n,).

    Returns:
        numpy.ndarray: eCADR in L/s, shape (n, n_countries).
    """
    return (
        annual_ecadr
        * lifespan[:, None]
        * (1 - essential_share)[None, :]
        * recovered[:, None]
    )


def ramp(weeks, start, length):
    """
    Fraction delivered by each week for a linear ramp.

    Arguments:
        weeks (int): Number of weeks to return.
        start (float): Week the ramp begins.
        length (float): Ramp duration in weeks.

    Returns:
        numpy.ndarray: Values from 0 to 1, shape (weeks,).
    """
    elapsed = np.arange(1, weeks + 1) - start
    if length <= 0:
        return np.clip(np.sign(elapsed) + 1, 0, 1).astype(float)
    return np.clip(elapsed / length, 0, 1)


def scenario_multiplier(
    samples, settings, n, baseline_filters, scenario, apply_cap=True
):
    """
    Weekly production multiplier relative to current output, by scenario.

    Scenario 1 is the utilisation increase alone. Scenario 3 is the COVID
    analogue growth. Scenario 2 is scenario 3 capped by the extra filters the
    available meltblown allowance supports.

    Arguments:
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.
        n (int): Number of draws.
        baseline_filters (numpy.ndarray): Current annual filter production, shape (n,).
        scenario (int): 1, 2 or 3.
        apply_cap (bool): Whether the scenario 2 meltblown cap applies. Coal
            baghouse bags are woven synthetics or glass, not meltblown
            polypropylene, so they are not capped.

    Returns:
        numpy.ndarray: Multiplier, shape (n, weeks).
    """
    weeks = int(settings["weeks"])
    delay = settings["start_delay_weeks"]

    utilisation = 1 + (1 / settings["utilisation_baseline"] - 1) * ramp(
        weeks, delay, settings["utilisation_ramp_weeks"]
    )
    if scenario == 1:
        return np.tile(utilisation, (n, 1))

    growth = 1 + (samples["scenario3_multiplier"][:, None] - 1) * ramp(
        weeks, delay, settings["scenario3_ramp_weeks"]
    )
    if scenario == 3 or not apply_cap:
        return growth

    grams_per_filter = (
        samples["filter_media_area_m2"]
        * samples["meltblown_layers_per_filter"]
        * samples["meltblown_basis_weight_gsm"]
    )
    grams_available = (
        samples["meltblown_total_tonnes"] * samples["meltblown_share_available"] * 1e6
    )
    extra_filters = grams_available / grams_per_filter
    cap = (1 + extra_filters / baseline_filters)[:, None]
    print(f"    meltblown cap allows a {np.median(cap):.1f}x increase (median)")
    return np.minimum(growth, cap)


def manufacturing_timeline(annual_ecadr, multiplier, settings):
    """
    Cumulative eCADR from a manufacturing channel, week by week.

    Weekly output is annual production divided by 52 and scaled by the scenario
    multiplier, plus the release of existing inventory.

    Arguments:
        annual_ecadr (numpy.ndarray): Annual eCADR, shape (n, n_countries).
        multiplier (numpy.ndarray): Weekly multiplier, shape (n, weeks).
        settings (dict): Fixed settings.

    Returns:
        numpy.ndarray: Cumulative eCADR, shape (weeks, n, n_countries).
    """
    weeks = int(settings["weeks"])
    delay = settings["start_delay_weeks"]
    active = (np.arange(1, weeks + 1) >= delay).astype(float)
    weekly = np.cumsum(multiplier * active / 52.0, axis=1)

    inventory = (
        settings["inventory_weeks"]
        / 52.0
        * ramp(weeks, delay, settings["inventory_release_weeks"])
    )
    years_delivered = weekly + inventory[None, :]
    return years_delivered.T[:, :, None] * annual_ecadr[None, :, :]


def one_off_timeline(total_ecadr, settings, start, length):
    """
    Cumulative eCADR from a one-off quantity spread over a number of weeks.

    Arguments:
        total_ecadr (numpy.ndarray): Total eCADR, shape (n, n_countries).
        settings (dict): Fixed settings.
        start (float): Week delivery begins.
        length (float): Weeks over which it is spread.

    Returns:
        numpy.ndarray: Cumulative eCADR, shape (weeks, n, n_countries).
    """
    weeks = int(settings["weeks"])
    fraction = ramp(weeks, start, length)
    return fraction[:, None, None] * total_ecadr[None, :, :]


def coverage(cumulative, requirement, df, interval):
    """
    Weekly share of a workforce requirement met, globally and by UN region.

    Coverage is not capped at one, so supply beyond a region's own requirement
    stays visible; figures cap the scale where that matters.

    Arguments:
        cumulative (numpy.ndarray): Cumulative eCADR, shape (weeks, n, n_countries).
        requirement (numpy.ndarray): eCADR requirement, shape (n_countries,).
        df (pandas.DataFrame): Country table with a region column.
        interval (float): Width of the reported uncertainty interval, in percent.

    Returns:
        pandas.DataFrame: Median and interval bounds by region and week.
    """
    weeks = cumulative.shape[0]
    tail = (100.0 - interval) / 2.0
    frames = []
    for region in sorted(df.region.unique()) + ["Global"]:
        mask = (
            np.ones(len(df), bool)
            if region == "Global"
            else (df.region == region).to_numpy()
        )
        share = cumulative[:, :, mask].sum(axis=2) / requirement[mask].sum()
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "week": np.arange(1, weeks + 1),
                    "coverage_median": np.median(share, axis=1),
                    "coverage_lower": np.percentile(share, tail, axis=1),
                    "coverage_upper": np.percentile(share, 100.0 - tail, axis=1),
                    "interval_percent": interval,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_streams(
    df, samples, settings, n, scenario, shares, prioritize_cr_boxes=False
):
    """
    Cumulative weekly eCADR for every supply channel under one scenario.

    Arguments:
        df (pandas.DataFrame): Country table.
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.
        n (int): Number of draws.
        scenario (int): 1, 2 or 3.
        shares (numpy.ndarray): Country share of global production.
        prioritize_cr_boxes (bool): If True, panel PAC units feed CR boxes
            instead of PACs. If False (the default), panel filters stay with
            PACs.

    Returns:
        dict: Channel name to cumulative eCADR, shape (weeks, n, n_countries).
    """
    delay = settings["start_delay_weeks"]
    essential = df["%Essential Workers"].to_numpy(float)

    total_revenue = samples["total_air_filter_revenue_usd"]
    panel_pac_deduction = (
        None if prioritize_cr_boxes else pac_panel_cr_box_filters(samples)
    )
    pac_global = pac_ecadr_global(samples, prioritize_cr_boxes=prioritize_cr_boxes)
    cr_global, baseline_filters, limiting = cr_box_ecadr_global(
        samples, settings, total_revenue, panel_pac_deduction=panel_pac_deduction
    )
    print(f"  CR box production is limited by {limiting}")

    pac_country = pac_global[:, None] * shares[None, :]
    cr_country = cr_global[:, None] * shares[None, :]

    # Equation 10 repurposes only filters in industrial settings
    industrial_ecadr, _, _ = cr_box_ecadr_global(
        samples, settings, industrial_revenue_total(samples)
    )
    industrial_country = industrial_ecadr[:, None] * shares[None, :]

    multiplier = scenario_multiplier(samples, settings, n, baseline_filters, scenario)

    streams = {
        "pac": manufacturing_timeline(pac_country, multiplier, settings),
        "cr_box": manufacturing_timeline(cr_country, multiplier, settings),
        "repurposed_pac": one_off_timeline(
            repurposed_ecadr(
                pac_country,
                samples["pac_lifespan_years"],
                essential,
                samples["fraction_recovered"],
            ),
            settings,
            delay,
            settings["repurposing_release_weeks"],
        ),
        "repurposed_cr_box": one_off_timeline(
            repurposed_ecadr(
                industrial_country,
                samples["panel_filter_lifespan_years"],
                essential,
                samples["fraction_recovered"],
            ),
            settings,
            delay,
            settings["repurposing_release_weeks"],
        ),
    }

    # Coal baghouse: annual replacement-bag production, plus bags recovered from
    # plants left idle by the fall in electricity demand
    standing = baghouse_standing_ecadr(df, samples, settings)
    bag_annual = standing / samples["baghouse_bag_life_years"]
    bag_repurposed = (
        standing * samples["idle_coal_capacity_share"] * samples["fraction_recovered"]
    )
    uncapped = scenario_multiplier(
        samples, settings, n, baseline_filters, scenario, apply_cap=False
    )
    streams["baghouse"] = manufacturing_timeline(
        bag_annual[:, None] * shares[None, :], uncapped, settings
    )
    streams["repurposed_baghouse"] = one_off_timeline(
        bag_repurposed[:, None] * shares[None, :],
        settings,
        delay,
        settings["baghouse_release_weeks"],
    )
    return streams


def write_requirements(df, results_dir=RESULTS_DIR):
    """
    Write the eCADR requirement of each UN region, and of the world.

    Figures divide supply by these totals, so writing them alongside the
    results keeps the denominator identical to the one the model itself used.

    Arguments:
        df (pandas.DataFrame): Country table.
        results_dir (str): Directory for the output CSV.
    """
    columns = [
        "Indoor Vital CADR Requirement (L/s)",
        "Indoor Essential CADR Requirement (L/s)",
    ]
    totals = df.groupby("region")[columns].sum()
    totals.loc["Global"] = totals.sum()
    totals.columns = ["indoor_vital_ecadr_l_per_s", "indoor_essential_ecadr_l_per_s"]
    totals.index.name = "region"
    totals.to_csv(os.path.join(results_dir, "requirements_by_region.csv"))


def write_results(df, streams, settings, scenario, results_dir=RESULTS_DIR):
    """
    Write weekly eCADR by country and channel, and workforce coverage.

    Per-channel eCADR is the median of each channel taken separately, so the
    channels sum to slightly more or less than the median total. That is the
    price of showing a composition, and the difference is small.

    Arguments:
        df (pandas.DataFrame): Country table.
        streams (dict): Output of build_streams.
        settings (dict): Fixed settings.
        scenario (int): 1, 2 or 3.
        results_dir (str): Directory for the output CSVs.
    """
    weeks = int(settings["weeks"])
    total = sum(streams.values())
    suffix = f"scenario{scenario}"

    weekly = pd.DataFrame(
        np.median(total, axis=1).T,
        index=df["Country Name"],
        columns=range(1, weeks + 1),
    )
    weekly.to_csv(os.path.join(results_dir, f"weekly_ecadr_by_country_{suffix}.csv"))

    pd.DataFrame(
        {name: np.median(streams[name].sum(axis=2), axis=1) for name in CHANNELS},
        index=pd.Index(range(1, weeks + 1), name="week"),
    ).to_csv(os.path.join(results_dir, f"ecadr_by_channel_{suffix}.csv"))

    for label, column in [
        ("vital", "Indoor Vital CADR Requirement (L/s)"),
        ("essential", "Indoor Essential CADR Requirement (L/s)"),
    ]:
        result = coverage(
            total,
            df[column].to_numpy(float),
            df,
            settings["uncertainty_interval"],
        )
        result.to_csv(
            os.path.join(results_dir, f"coverage_{label}_{suffix}.csv"), index=False
        )
        global_rows = result[result.region == "Global"].set_index("week")
        for week in REPORT_WEEKS:
            print(
                f"    {label} workers covered globally at week {week:>2}: "
                f"{global_rows.loc[week, 'coverage_median']:.1%}"
            )


def run_scale_up(df, samples, settings, n, shares, results_dir, prioritize_cr_boxes):
    """
    Run all three scenarios and write results to one directory.

    Arguments:
        df (pandas.DataFrame): Country table.
        samples (dict): Output of sample_all.
        settings (dict): Fixed settings.
        n (int): Number of draws.
        shares (numpy.ndarray): Country share of global production.
        results_dir (str): Where to write the CSVs.
        prioritize_cr_boxes (bool): Whether panel PAC units feed CR boxes.
    """
    os.makedirs(results_dir, exist_ok=True)
    write_requirements(df, results_dir=results_dir)
    for scenario, description in SCENARIOS.items():
        print(f"\nScenario {scenario}: {description}")
        streams = build_streams(
            df,
            samples,
            settings,
            n,
            scenario,
            shares,
            prioritize_cr_boxes=prioritize_cr_boxes,
        )
        write_results(df, streams, settings, scenario, results_dir=results_dir)


def main():
    print("=" * 70)
    print("IN-ROOM FILTRATION SCALE-UP")
    print("=" * 70)

    params = load_parameters()
    settings = load_settings()
    np.random.seed(int(settings["random_seed"]))
    n = int(settings["n_draws"])
    samples = sample_all(params, n)

    df = load_country_data(bool(settings.get("adjust_MVA_by_cost")))

    print("\nAllocating global production across countries...")
    total_filters = filter_production(
        samples, samples["total_air_filter_revenue_usd"], MERV_BANDS
    )
    print(f"  total panel filter production: {np.median(total_filters):,.0f} per year")
    shares = production_shares(
        df,
        settings["mva_exponent_b"],
        settings["min_country_filter_production"],
        total_filters.mean(),
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pacs_prioritized_dir = os.path.join(RESULTS_DIR, "PACs_prioritized")
    cr_boxes_prioritized_dir = os.path.join(RESULTS_DIR, "CR_boxes_prioritized")

    print("\nPACs prioritized (panel filters stay with PACs)...")
    run_scale_up(
        df,
        samples,
        settings,
        n,
        shares,
        pacs_prioritized_dir,
        prioritize_cr_boxes=False,
    )

    print("\nCR boxes prioritized (panel filters diverted from PACs)...")
    run_scale_up(
        df,
        samples,
        settings,
        n,
        shares,
        cr_boxes_prioritized_dir,
        prioritize_cr_boxes=True,
    )

    print(
        f"\nResults written to {pacs_prioritized_dir}/ and {cr_boxes_prioritized_dir}/"
    )


if __name__ == "__main__":
    main()
