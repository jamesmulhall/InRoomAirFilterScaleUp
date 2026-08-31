"""Tests for the scale-up model in ``src/scale_up_model.py``.

These cover the parts of the model that do not need a full run, so they
stay useful while the parameters that are still missing from the methods
are filled in. The model refuses to run until those are set, and the
tests below check that refusal as well as the maths around it.

The fitted inputs in ``src/linear_models.py`` are tested here too, since
the model reads their output from ``data/scale_up/settings.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import linear_models as lm
import scale_up_model as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PARAMETERS = REPO_ROOT / "data" / "scale_up" / "parameters.csv"
REAL_SETTINGS = REPO_ROOT / "data" / "scale_up" / "settings.csv"
REAL_ALLOCATOR = REPO_ROOT / "data" / "scale_up" / "allocator_fit_data.csv"
REAL_COAL = REPO_ROOT / "data" / "scale_up" / "coal_plant_airflow.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parameters(path, rows):
    """
    Write a small parameter table in the format load_parameters expects.

    Arguments:
        path (pathlib.Path): Where to write the CSV.
        rows (list): Dicts with parameter, low, high and distribution.

    Returns:
        str: The path written, as a string.
    """
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _settings(**overrides):
    """
    Fixed settings for the timeline functions, with overrides applied.

    Arguments:
        **overrides: Settings to replace.

    Returns:
        dict: Setting name to value.
    """
    settings = {
        "weeks": 10,
        "start_delay_weeks": 2,
        "utilisation_baseline": 0.75,
        "utilisation_ramp_weeks": 4,
        "scenario3_ramp_weeks": 4,
    }
    settings.update(overrides)
    return settings


def _samples(n=3, **overrides):
    """
    Draws for the scenario functions, with overrides applied.

    Arguments:
        n (int): Number of draws.
        **overrides: Sample arrays to replace.

    Returns:
        dict: Parameter name to samples, each shape (n,).
    """
    samples = {
        "scenario3_multiplier": np.full(n, 4.0),
        "filter_media_area_m2": np.full(n, 0.47),
        "meltblown_layers_per_filter": np.full(n, 1.0),
        "meltblown_basis_weight_gsm": np.full(n, 25.0),
        "meltblown_total_tonnes": np.full(n, 253500.0),
        "meltblown_share_available": np.full(n, 0.12),
    }
    samples.update(overrides)
    return samples


class _FakeResponse:
    """One page of a World Bank API response."""

    def __init__(self, pages, records):
        self._payload = [{"pages": pages}, records]

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# load_parameters: fail loudly while values are missing
# ---------------------------------------------------------------------------


def test_load_parameters_names_every_unset_parameter(tmp_path):
    """Parameters left at low == high == 0 are all listed in the error."""
    path = _write_parameters(
        tmp_path / "parameters.csv",
        [
            {"parameter": "filled", "low": 1, "high": 2, "distribution": "normal"},
            {"parameter": "missing_one", "low": 0, "high": 0, "distribution": "normal"},
            {"parameter": "missing_two", "low": 0, "high": 0, "distribution": "normal"},
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        sm.load_parameters(path)
    message = str(excinfo.value)
    assert "missing_one" in message
    assert "missing_two" in message
    assert "filled" not in message


def test_load_parameters_accepts_a_filled_table(tmp_path):
    """A table with no unset rows loads and is indexed by parameter."""
    path = _write_parameters(
        tmp_path / "parameters.csv",
        [
            {"parameter": "fixed", "low": 5, "high": 5, "distribution": "normal"},
            {"parameter": "ranged", "low": 1, "high": 3, "distribution": "lognormal"},
        ],
    )
    params = sm.load_parameters(path)
    assert list(params.index) == ["fixed", "ranged"]
    assert params.loc["ranged", "high"] == 3


def test_real_parameter_table_matches_its_own_unset_rows():
    """The real table either loads, or refuses and names its unset rows.

    This holds both before and after the parameters that are missing from
    the methods are filled in.
    """
    table = pd.read_csv(REAL_PARAMETERS).set_index("parameter")
    unset = table[(table.low == 0) & (table.high == 0)].index.tolist()
    if not unset:
        assert len(sm.load_parameters(REAL_PARAMETERS)) == len(table)
        return
    with pytest.raises(ValueError) as excinfo:
        sm.load_parameters(REAL_PARAMETERS)
    for name in unset:
        assert name in str(excinfo.value)


def test_real_parameter_distributions_are_all_supported():
    """Every distribution named in the real table can be sampled."""
    table = pd.read_csv(REAL_PARAMETERS)
    assert set(table.distribution) <= {"normal", "lognormal", "uniform"}


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------


def test_load_settings_reads_numbers_as_floats(tmp_path):
    """Numeric settings become floats; anything else stays a string."""
    path = tmp_path / "settings.csv"
    pd.DataFrame(
        [
            {"setting": "weeks", "value": "26"},
            {"setting": "label", "value": "coal baghouse"},
        ]
    ).to_csv(path, index=False)
    settings = sm.load_settings(str(path))
    assert settings["weeks"] == pytest.approx(26.0)
    assert settings["label"] == "coal baghouse"


def test_real_settings_cover_everything_the_model_reads():
    """Every setting the model looks up is present in settings.csv."""
    settings = sm.load_settings(REAL_SETTINGS)
    required = [
        "n_draws",
        "weeks",
        "random_seed",
        "start_delay_weeks",
        "utilisation_baseline",
        "utilisation_ramp_weeks",
        "inventory_weeks",
        "inventory_release_weeks",
        "repurposing_release_weeks",
        "baghouse_release_weeks",
        "baghouse_gradient",
        "baghouse_intercept_l_per_s",
        "mva_exponent_b",
        "min_country_filter_production",
        "filters_per_cr_box",
        "pc_fans_per_cr_box",
        "scenario3_ramp_weeks",
        "uncertainty_interval",
    ]
    missing = [name for name in required if name not in settings]
    assert not missing, f"settings.csv is missing: {missing}"


# ---------------------------------------------------------------------------
# sample_all
# ---------------------------------------------------------------------------


def test_sample_all_returns_one_array_per_parameter(tmp_path):
    """Every parameter gets n draws, and low == high gives a constant."""
    path = _write_parameters(
        tmp_path / "parameters.csv",
        [
            {"parameter": "fixed", "low": 7, "high": 7, "distribution": "normal"},
            {"parameter": "gauss", "low": 1, "high": 3, "distribution": "normal"},
            {"parameter": "logn", "low": 1, "high": 3, "distribution": "lognormal"},
            {"parameter": "unif", "low": 1, "high": 3, "distribution": "uniform"},
        ],
    )
    samples = sm.sample_all(sm.load_parameters(path), n=500)
    assert set(samples) == {"fixed", "gauss", "logn", "unif"}
    for name, draws in samples.items():
        assert draws.shape == (500,), name
    assert np.all(samples["fixed"] == 7)
    assert np.all(samples["logn"] > 0)
    assert samples["unif"].min() >= 1
    assert samples["unif"].max() <= 3
    assert samples["gauss"].mean() == pytest.approx(2.0, abs=0.2)


def test_sample_all_rejects_an_unknown_distribution(tmp_path):
    """An unsupported distribution name is an error, not a silent default."""
    path = _write_parameters(
        tmp_path / "parameters.csv",
        [{"parameter": "odd", "low": 1, "high": 2, "distribution": "triangular"}],
    )
    with pytest.raises(ValueError, match="triangular"):
        sm.sample_all(sm.load_parameters(path), n=10)


def test_sample_all_is_reproducible_under_a_seed(tmp_path):
    """The same seed gives the same draws, which is what lets scenarios be
    compared draw for draw."""
    path = _write_parameters(
        tmp_path / "parameters.csv",
        [{"parameter": "gauss", "low": 1, "high": 3, "distribution": "normal"}],
    )
    params = sm.load_parameters(path)
    np.random.seed(0)
    first = sm.sample_all(params, n=50)["gauss"]
    np.random.seed(0)
    second = sm.sample_all(params, n=50)["gauss"]
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# production_shares
# ---------------------------------------------------------------------------


def _country_table(mva):
    return pd.DataFrame({"mva_usd": mva})


def test_production_shares_are_proportional_to_mva_when_exponent_is_one():
    """With b = 1 the allocation is simple proportionality."""
    df = _country_table([100.0, 300.0])
    shares = sm.production_shares(df, exponent=1.0, min_production=0, total_filters=1e9)
    np.testing.assert_allclose(shares, [0.25, 0.75])


def test_production_shares_exclude_countries_without_mva():
    """A country with no reported MVA cannot manufacture."""
    df = _country_table([100.0, 0.0, 100.0])
    shares = sm.production_shares(df, exponent=1.0, min_production=0, total_filters=1e9)
    assert shares[1] == 0
    assert shares.sum() == pytest.approx(1.0)


def test_production_shares_reallocate_below_the_threshold():
    """Countries under the minimum are zeroed and their share moves to the rest."""
    df = _country_table([1e12, 1e12, 1.0])
    shares = sm.production_shares(
        df, exponent=1.0, min_production=1000, total_filters=1e6
    )
    assert shares[2] == 0
    assert shares.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(shares[:2], [0.5, 0.5])


def test_production_shares_apply_the_exponent():
    """The exponent bends the allocation away from proportionality."""
    df = _country_table([10.0, 100.0])
    shares = sm.production_shares(df, exponent=2.0, min_production=0, total_filters=1e9)
    np.testing.assert_allclose(shares, [100 / 10100, 10000 / 10100])


# ---------------------------------------------------------------------------
# ramp
# ---------------------------------------------------------------------------


def test_ramp_rises_linearly_between_start_and_start_plus_length():
    values = sm.ramp(weeks=8, start=2, length=4)
    np.testing.assert_allclose(values, [0, 0, 0.25, 0.5, 0.75, 1, 1, 1])


def test_ramp_with_zero_length_is_a_step():
    """A zero-length ramp switches on at the start week."""
    values = sm.ramp(weeks=5, start=3, length=0)
    np.testing.assert_allclose(values, [0, 0, 1, 1, 1])


def test_ramp_stays_at_zero_before_a_late_start():
    values = sm.ramp(weeks=4, start=10, length=4)
    np.testing.assert_allclose(values, np.zeros(4))


# ---------------------------------------------------------------------------
# Global production channels
# ---------------------------------------------------------------------------


def test_pac_ecadr_global_multiplies_the_eligible_share_by_unit_ecadr():
    """Equations 1 and 2: eligible revenue over price, times eCADR per unit."""
    samples = {
        "pac_market_revenue_usd": np.array([1000.0]),
        "fraction_room": np.array([0.5]),
        "fraction_merv13_plus": np.array([0.5]),
        "fraction_non_panel": np.array([0.5]),
        "pac_price_usd": np.array([25.0]),
        "pac_ecadr_l_per_s": np.array([100.0]),
    }
    # 1000 * 0.5 * 0.5 * 0.5 / 25 = 5 units, at 100 L/s each
    np.testing.assert_allclose(sm.pac_ecadr_global(samples), [500.0])


def _band_samples(**overrides):
    """Equal revenue and price in every MERV band, so shares are 1/5 each."""
    samples = {f"industrial_revenue_{b}_usd": np.array([100.0]) for b in sm.MERV_BANDS}
    samples.update({f"price_per_volume_{b}": np.array([1000.0]) for b in sm.MERV_BANDS})
    samples["fraction_panel"] = np.array([0.5])
    samples["filter_volume_m3"] = np.array([0.01])
    samples.update(overrides)
    return samples


def test_merv_revenue_shares_default_to_the_merv13_bands():
    """By default only MERV 13-16 and 17-20 are returned, as market shares."""
    shares = sm.merv_revenue_shares(_band_samples())
    assert set(shares) == set(sm.MERV13_BANDS)
    for band in shares.values():
        np.testing.assert_allclose(band, [0.2])


def test_merv_revenue_shares_over_all_bands_sum_to_one():
    shares = sm.merv_revenue_shares(_band_samples(), sm.MERV_BANDS)
    assert set(shares) == set(sm.MERV_BANDS)
    np.testing.assert_allclose(sum(shares.values()), [1.0])


def test_filter_production_divides_by_price_times_volume():
    """Equation 4 divides revenue by price per cubic metre times filter volume."""
    # Each of the two bands: 1000 * 0.2 * 0.5 / (1000 * 0.01) = 10 filters
    filters = sm.filter_production(_band_samples(), np.array([1000.0]))
    np.testing.assert_allclose(filters, [20.0])


def test_filter_production_over_all_bands_is_the_total():
    """All five bands give total production, and MERV 13+ is a part of it."""
    samples, revenue = _band_samples(), np.array([1000.0])
    total = sm.filter_production(samples, revenue, sm.MERV_BANDS)
    merv13 = sm.filter_production(samples, revenue, sm.MERV13_BANDS)
    np.testing.assert_allclose(total, [50.0])
    np.testing.assert_allclose(merv13 / total, [0.4])


def test_industrial_revenue_total_sums_the_bands():
    total = sm.industrial_revenue_total(_band_samples())
    np.testing.assert_allclose(total, [500.0])


def test_every_merv_band_has_a_price_in_the_parameter_table():
    """Total production needs a price for all five bands, not just MERV 13+."""
    parameters = pd.read_csv(REAL_PARAMETERS).parameter.tolist()
    missing = [
        f"price_per_volume_{band}"
        for band in sm.MERV_BANDS
        if f"price_per_volume_{band}" not in parameters
    ]
    assert not missing, f"parameters.csv is missing: {missing}"


def test_total_filter_production_is_derived_not_a_parameter():
    """The threshold total comes from equation 4, so it cannot drift from it."""
    parameters = pd.read_csv(REAL_PARAMETERS).parameter.tolist()
    assert "total_filter_production_global" not in parameters


# ---------------------------------------------------------------------------
# scenario_multiplier
# ---------------------------------------------------------------------------


def test_scenario_one_ramps_utilisation_to_full_capacity():
    """Scenario 1 is the utilisation increase alone: 1 -> 1 / 0.75."""
    multiplier = sm.scenario_multiplier(
        _samples(), _settings(), n=3, baseline_filters=np.full(3, 1e9), scenario=1
    )
    assert multiplier.shape == (3, 10)
    np.testing.assert_allclose(multiplier[:, 0], 1.0)
    np.testing.assert_allclose(multiplier[:, -1], 1 / 0.75)
    assert np.all(np.diff(multiplier, axis=1) >= 0)


def test_scenario_three_ramps_to_the_covid_analogue_growth():
    """Scenario 3 reaches the sampled growth multiplier."""
    multiplier = sm.scenario_multiplier(
        _samples(), _settings(), n=3, baseline_filters=np.full(3, 1e9), scenario=3
    )
    np.testing.assert_allclose(multiplier[:, -1], 4.0)


def test_scenario_two_never_exceeds_scenario_three():
    """Scenario 2 is scenario 3 capped by the meltblown allowance."""
    settings, samples = _settings(), _samples()
    baseline = np.full(3, 1e9)
    capped = sm.scenario_multiplier(samples, settings, 3, baseline, scenario=2)
    uncapped = sm.scenario_multiplier(samples, settings, 3, baseline, scenario=3)
    assert np.all(capped <= uncapped + 1e-12)


def test_scenario_two_binds_when_meltblown_is_scarce():
    """A small meltblown allowance holds growth below scenario 3."""
    settings = _settings()
    samples = _samples(meltblown_total_tonnes=np.full(3, 1.0))
    baseline = np.full(3, 1e9)
    capped = sm.scenario_multiplier(samples, settings, 3, baseline, scenario=2)
    uncapped = sm.scenario_multiplier(samples, settings, 3, baseline, scenario=3)
    assert capped[:, -1].max() < uncapped[:, -1].min()


def test_scenario_two_without_the_cap_equals_scenario_three():
    """The coal baghouse channel is not meltblown, so it is not capped."""
    settings, samples = _settings(), _samples()
    baseline = np.full(3, 1e9)
    uncapped = sm.scenario_multiplier(
        samples, settings, 3, baseline, scenario=2, apply_cap=False
    )
    scenario3 = sm.scenario_multiplier(samples, settings, 3, baseline, scenario=3)
    np.testing.assert_allclose(uncapped, scenario3)


# ---------------------------------------------------------------------------
# coverage and requirements
# ---------------------------------------------------------------------------


def _coverage_inputs(weeks=4, n=100):
    """
    A small country table and cumulative eCADR array for coverage().

    Draw i supplies i units of eCADR to every country every week, so the
    spread of coverage across draws is known exactly.

    Arguments:
        weeks (int): Weeks in the timeline.
        n (int): Number of draws.

    Returns:
        tuple: (cumulative eCADR, requirement, country table).
    """
    df = pd.DataFrame(
        {
            "Country Name": ["A", "B", "C"],
            "region": ["North", "North", "South"],
        }
    )
    draws = np.arange(1, n + 1, dtype=float)
    cumulative = np.tile(draws[None, :, None], (weeks, 1, len(df)))
    requirement = np.array([1.0, 1.0, 2.0])
    return cumulative, requirement, df


def test_coverage_reports_every_week_for_every_region():
    """Each UN region present, plus Global, at every week of the timeline."""
    cumulative, requirement, df = _coverage_inputs(weeks=4)
    result = sm.coverage(cumulative, requirement, df, interval=90)
    assert set(result.region) == {"North", "South", "Global"}
    assert sorted(result[result.region == "Global"].week) == [1, 2, 3, 4]
    assert len(result) == 12


def test_coverage_is_not_capped_at_full_coverage():
    """Supply beyond the requirement stays visible above 100 percent."""
    cumulative, requirement, df = _coverage_inputs()
    result = sm.coverage(cumulative, requirement, df, interval=90)
    assert result.coverage_median.min() > 1.0


def test_coverage_interval_bounds_follow_the_setting():
    """The bounds are the tails the requested interval implies."""
    cumulative, requirement, df = _coverage_inputs(n=100)
    draws = np.arange(1, 101, dtype=float)

    wide = sm.coverage(cumulative, requirement, df, interval=90).iloc[0]
    assert wide.coverage_lower == pytest.approx(np.percentile(draws, 5))
    assert wide.coverage_upper == pytest.approx(np.percentile(draws, 95))
    assert wide.interval_percent == 90

    narrow = sm.coverage(cumulative, requirement, df, interval=50).iloc[0]
    assert narrow.coverage_lower > wide.coverage_lower
    assert narrow.coverage_upper < wide.coverage_upper


def test_write_requirements_totals_each_region_and_the_world(tmp_path, monkeypatch):
    """Regional requirements sum the countries, and Global sums the regions."""
    monkeypatch.setattr(sm, "RESULTS_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "region": ["North", "North", "South"],
            "Indoor Vital CADR Requirement (L/s)": [1.0, 2.0, 4.0],
            "Indoor Essential CADR Requirement (L/s)": [10.0, 20.0, 40.0],
        }
    )
    sm.write_requirements(df)

    totals = pd.read_csv(tmp_path / "requirements_by_region.csv", index_col="region")
    assert totals.loc["North", "indoor_vital_ecadr_l_per_s"] == 3.0
    assert totals.loc["Global", "indoor_vital_ecadr_l_per_s"] == 7.0
    assert totals.loc["Global", "indoor_essential_ecadr_l_per_s"] == 70.0


# ---------------------------------------------------------------------------
# fetch_mva
# ---------------------------------------------------------------------------


def test_fetch_mva_takes_the_most_recent_positive_year(tmp_path, monkeypatch):
    """Pagination, nulls, zeros and blank ISO3 codes are all handled."""
    pages = {
        1: [
            {"countryiso3code": "AAA", "date": "2015", "value": 100.0},
            {"countryiso3code": "AAA", "date": "2020", "value": 300.0},
            {"countryiso3code": "BBB", "date": "2019", "value": None},
        ],
        2: [
            {"countryiso3code": "BBB", "date": "2014", "value": 50.0},
            {"countryiso3code": "CCC", "date": "2018", "value": 0.0},
            {"countryiso3code": "", "date": "2018", "value": 900.0},
        ],
    }
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["page"])
        return _FakeResponse(pages=2, records=pages[params["page"]])

    monkeypatch.setattr(sm.requests, "get", fake_get)

    cache = tmp_path / "cache" / "mva.csv"
    df = sm.fetch_mva(cache=str(cache), years="2010:2024")

    assert calls == [1, 2]
    assert set(df.iso3) == {"AAA", "BBB"}
    row = df[df.iso3 == "AAA"].iloc[0]
    assert row.mva_usd == 300.0
    assert row.mva_year == 2020
    assert cache.exists()


def test_fetch_mva_uses_the_cache_without_downloading(tmp_path, monkeypatch):
    """Once cached, the model runs offline."""
    cache = tmp_path / "mva.csv"
    pd.DataFrame([{"iso3": "AAA", "mva_year": 2020, "mva_usd": 300.0}]).to_csv(
        cache, index=False
    )

    def fail_get(*args, **kwargs):
        raise AssertionError("fetch_mva downloaded despite a cache being present")

    monkeypatch.setattr(sm.requests, "get", fail_get)
    df = sm.fetch_mva(cache=str(cache))
    assert list(df.iso3) == ["AAA"]


def test_fetch_mva_refresh_ignores_the_cache(tmp_path, monkeypatch):
    """refresh=True re-downloads over an existing cache."""
    cache = tmp_path / "mva.csv"
    pd.DataFrame([{"iso3": "OLD", "mva_year": 2011, "mva_usd": 1.0}]).to_csv(
        cache, index=False
    )

    monkeypatch.setattr(
        sm.requests,
        "get",
        lambda url, params, timeout: _FakeResponse(
            pages=1,
            records=[{"countryiso3code": "NEW", "date": "2021", "value": 5.0}],
        ),
    )
    df = sm.fetch_mva(cache=str(cache), refresh=True)
    assert list(df.iso3) == ["NEW"]


# ---------------------------------------------------------------------------
# linear_models
# ---------------------------------------------------------------------------


def test_fit_coal_airflow_refuses_an_empty_sample(tmp_path):
    """The coal fit fails loudly rather than fitting nothing."""
    path = tmp_path / "coal_plant_airflow.csv"
    pd.DataFrame(columns=["plant", "capacity_mw", "airflow_l_per_s"]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="empty"):
        lm.fit_coal_airflow(str(path))


def test_fit_coal_airflow_recovers_a_known_line(tmp_path):
    """Airflow = 1000 x MW + 5000 is recovered exactly from points on it."""
    path = tmp_path / "coal_plant_airflow.csv"
    capacity = np.array([100.0, 200.0, 300.0])
    pd.DataFrame(
        {
            "plant": ["a", "b", "c"],
            "capacity_mw": capacity,
            "airflow_l_per_s": 1000 * capacity + 5000,
        }
    ).to_csv(path, index=False)
    fit = lm.fit_coal_airflow(str(path))
    assert fit["slope"] == pytest.approx(1000.0)
    assert fit["intercept"] == pytest.approx(5000.0)
    assert fit["n"] == 3


def test_real_coal_sample_is_either_empty_or_fittable():
    """Documents that the coal sample still needs the Supplementary Information."""
    df = pd.read_csv(REAL_COAL)
    if df.empty:
        with pytest.raises(ValueError, match="empty"):
            lm.fit_coal_airflow(str(REAL_COAL))
        return
    fit = lm.fit_coal_airflow(str(REAL_COAL))
    assert np.isfinite(fit["slope"])
    assert np.isfinite(fit["intercept"])


def test_fit_allocator_reproduces_the_mva_exponent():
    """The pooled slope is the exponent b the model allocates production with."""
    fit = lm.fit_allocator(str(REAL_ALLOCATOR))
    pooled = fit["pooled"]
    slope = np.asarray(pooled.params)[1]
    assert int(pooled.nobs) == 40
    assert slope == pytest.approx(1.019, abs=0.001)
    assert pooled.rsquared > 0.85
    # Indistinguishable from simple proportionality
    assert abs(slope - 1) < 2 * np.asarray(pooled.bse)[1]


def test_settings_mva_exponent_matches_the_fit():
    """settings.csv must carry the exponent that linear_models.py fits."""
    fit = lm.fit_allocator(str(REAL_ALLOCATOR))
    fitted = round(np.asarray(fit["pooled"].params)[1], 3)
    assert sm.load_settings(REAL_SETTINGS)["mva_exponent_b"] == pytest.approx(fitted)


def test_update_settings_rewrites_only_the_fitted_rows(tmp_path):
    """The fitted values land in settings.csv without disturbing the rest."""
    path = tmp_path / "settings.csv"
    original = pd.read_csv(REAL_SETTINGS)
    original.to_csv(path, index=False)

    coal = {"slope": 1500.0, "intercept": 40000.0, "r_squared": 0.9, "n": 11}
    allocator = lm.fit_allocator(str(REAL_ALLOCATOR))
    written = lm.update_settings(coal, allocator, str(path))

    updated = sm.load_settings(str(path))
    assert updated["baghouse_gradient"] == pytest.approx(1500.0)
    assert updated["baghouse_intercept_l_per_s"] == pytest.approx(40000.0)
    assert updated["mva_exponent_b"] == pytest.approx(1.019, abs=0.001)
    assert list(written.setting) == [
        "baghouse_gradient",
        "baghouse_intercept_l_per_s",
        "mva_exponent_b",
    ]

    # Row order is kept, and settings the fits do not touch are unchanged
    after = pd.read_csv(path)
    assert list(after.setting) == list(original.setting)
    untouched = ~after.setting.isin(written.setting)
    pd.testing.assert_frame_equal(
        after[untouched].reset_index(drop=True),
        original[untouched.to_numpy()].reset_index(drop=True),
    )


def test_update_settings_records_the_provenance_of_each_fit(tmp_path):
    """Each fitted row carries the sample size and input file it came from."""
    path = tmp_path / "settings.csv"
    pd.read_csv(REAL_SETTINGS).to_csv(path, index=False)
    coal = {"slope": 1500.0, "intercept": 40000.0, "r_squared": 0.9, "n": 11}
    lm.update_settings(coal, lm.fit_allocator(str(REAL_ALLOCATOR)), str(path))

    rows = pd.read_csv(path).set_index("setting")
    assert "n = 11" in rows.loc["baghouse_gradient", "note"]
    assert rows.loc["baghouse_gradient", "source"] == lm.COAL_FILE
    assert "n = 40" in rows.loc["mva_exponent_b", "note"]
    assert rows.loc["mva_exponent_b", "source"] == lm.ALLOCATOR_FILE
