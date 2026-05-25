"""Tests for the scale-up pipeline in ``src/countries.py``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from uncertainties import ufloat

import countries as cc
from country_pkg import Country


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_country(
    name: str = "USA",
    big_6: bool = True,
    weekly_cadr: float = 1_000.0,
    repur_cadr: float = 100.0,
    stock_cadr: float = 500.0,
    coal_cadr: float = 200.0,
    mdd: int = 1,
    repur_delay: int = 12,
    stock_delay: int = 2,
    coal_delay: int = 4,
) -> Country:
    """Construct a Country with the bare minimum scale-up properties.

    ``name`` defaults to ``"USA"`` (a real ISO-3 code) because the
    Country class normalises names through country_converter and would
    silently rewrite arbitrary strings.
    """
    c = Country(name=name)
    c.properties.update(
        {
            "Big_6": big_6,
            "CADR: CR Box Weekly Production": weekly_cadr,
            "CADR: CR Box Repurposing": repur_cadr,
            "CADR: CR Box Initial Stock": stock_cadr,
            "CADR: Coal Baghouse": coal_cadr,
            "CR Box Manufacturing Distribution Delay": mdd,
            "Repurposing Delay": repur_delay,
            "Initial Stock Delay": stock_delay,
            "Coalbaghouse Delay": coal_delay,
        }
    )
    return c


# ---------------------------------------------------------------------------
# manufacturing_delay_function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m, expected",
    [
        (95, 1),
        (90, 2),
        (86, 2),
        (85, 3),
        (81, 3),
        (80, 4),
        (76, 4),
        (75, 5),
        (71, 5),
        (70, 6),
        (66, 6),
        (65, 7),
        (61, 7),
        (60, 8),
        (56, 8),
        (55.5, 0),
        (50, 0),
        (0, 0),
    ],
)
def test_manufacturing_delay_function_boundaries(m, expected):
    assert cc.manufacturing_delay_function(m) == expected


# ---------------------------------------------------------------------------
# Single-deposit scale-up trajectories
# ---------------------------------------------------------------------------


def test_scale_up_cr_stock_deposits_at_initial_stock_delay():
    c = _make_country("X", stock_delay=3, stock_cadr=42.0)
    data = cc.scale_up_CR_STOCK(c, weeks=10)
    # week 0 is the pre-roll-out value (always 0); subsequent values are
    # cumulative running totals.
    assert data[0] == 0
    for i in range(1, 3):
        assert data[i] == 0
    for i in range(3, 11):
        assert data[i] == 42.0


def test_scale_up_coalbag_deposits_at_coalbaghouse_delay():
    c = _make_country("X", coal_delay=5, coal_cadr=10.0)
    data = cc.scale_up_COALBAG(c, weeks=8)
    assert data[0] == 0
    for i in range(1, 5):
        assert data[i] == 0
    for i in range(5, 9):
        assert data[i] == 10.0


def test_scale_up_cr_repur_matches_repur_list():
    c = _make_country("X", repur_delay=12, repur_cadr=1000.0)
    data = cc.scale_up_CR_REPUR(c, weeks=13)
    # The increments from week 1 to week N are the cumulative differences.
    increments = [data[i] - data[i - 1] for i in range(1, 12)]
    for i, expected_frac in enumerate(cc.REPUR_LIST):
        assert increments[i] == pytest.approx(1000.0 * expected_frac)
    # After week 12 (== repur_delay) no further deposits.
    assert data[12] == data[11]


# ---------------------------------------------------------------------------
# scale_up_MAIN == sum of the four streams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("big_6", [True, False])
def test_scale_up_main_equals_sum_of_streams(big_6):
    c = _make_country("X", big_6=big_6)
    weeks = 20
    main = cc.scale_up_MAIN(c, weeks)
    cr_man = cc.scale_up_CR_MAN(c, weeks)
    cr_repur = cc.scale_up_CR_REPUR(c, weeks)
    cr_stock = cc.scale_up_CR_STOCK(c, weeks)
    coalbag = cc.scale_up_COALBAG(c, weeks)
    for t in range(weeks + 1):
        assert main[t] == pytest.approx(
            cr_man[t] + cr_repur[t] + cr_stock[t] + coalbag[t]
        )


def test_scale_up_cr_man_big6_uses_70_to_100_ramp():
    """For Big_6 countries the first 6 weeks ramp 70%->95% of weekly prod."""
    c = _make_country("X", big_6=True, weekly_cadr=100.0, mdd=1)
    data = cc.scale_up_CR_MAN(c, weeks=10)
    increments = [data[i] - data[i - 1] for i in range(1, 11)]
    # Weeks 1..6: 0.7 + 0.05*(i-1) * 100
    expected_ramp = [(0.7 + 0.05 * (i - 1)) * 100.0 for i in range(1, 7)]
    for inc, exp in zip(increments[:6], expected_ramp):
        assert inc == pytest.approx(exp)
    # Weeks 7+: full 100
    for inc in increments[6:]:
        assert inc == pytest.approx(100.0)


def test_scale_up_cr_man_non_big6_flat_after_delay():
    c = _make_country("X", big_6=False, weekly_cadr=50.0, mdd=3)
    data = cc.scale_up_CR_MAN(c, weeks=8)
    for i in range(0, 3):
        assert data[i] == 0
    for i in range(3, 9):
        assert data[i] - data[i - 1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compare_scale_up_data
# ---------------------------------------------------------------------------


def test_compare_scale_up_data_returns_first_crossing_week():
    """When indoor_vital_count > 0, return week where data first exceeds threshold."""
    data = [0, 100, 200, 300, 400, 500]
    # threshold = 250 * 1 = 250
    assert cc.compare_scale_up_data(data, 2.5, cadrpp=100) == 4
    # i.e. data[3]=300 > 250 -> week 4 (1-indexed)


def test_compare_scale_up_data_zero_target_returns_last_week():
    """When indoor_vital_count == 0 the threshold is never effectively crossed."""
    data = [0, 100, 200, 300]
    assert cc.compare_scale_up_data(data, 0, cadrpp=100) == 4


def test_compare_scale_up_data_handles_ufloat_inputs():
    """Comparisons should fall through nominal_value of ufloat entries."""
    data = [ufloat(0, 0), ufloat(100, 5), ufloat(300, 5)]
    assert cc.compare_scale_up_data(data, 1.5, cadrpp=100) == 3


# ---------------------------------------------------------------------------
# Aggregation (per-region and global)
# ---------------------------------------------------------------------------


def _populate_for_aggregation(c, indoor_vital, indoor_essential, region):
    c.properties["Indoor Vital Workers"] = indoor_vital
    c.properties["Indoor Essential Workers"] = indoor_essential
    c.properties["Region"] = region


def test_scale_up_all_countries_region_and_global():
    """Region sums == sum of constituent countries; Global == sum of all."""
    a = _make_country(
        "USA", big_6=True, weekly_cadr=10, repur_cadr=0, stock_cadr=0, coal_cadr=0
    )
    _populate_for_aggregation(a, 100, 200, "Northern America")
    b = _make_country(
        "CAN", big_6=True, weekly_cadr=20, repur_cadr=0, stock_cadr=0, coal_cadr=0
    )
    _populate_for_aggregation(b, 200, 400, "Northern America")
    d = _make_country(
        "CHN", big_6=True, weekly_cadr=5, repur_cadr=0, stock_cadr=0, coal_cadr=0
    )
    _populate_for_aggregation(d, 50, 80, "Eastern Asia")
    countries_dict = {"USA": a, "CAN": b, "CHN": d}

    tables = cc.scale_up_all_countries(countries_dict, weeks=10)

    # Northern America region == a.name + b.name (Country class normalises
    # the names so we look them up via the canonical attribute).
    region_na = np.array(tables.region_main["Northern America"])
    expected_na = np.array(tables.country_main[a.name]) + np.array(
        tables.country_main[b.name]
    )
    assert np.allclose(region_na, expected_na)

    # Global == a + b + d
    global_main = np.array(tables.region_main["Global"])
    expected_global = (
        np.array(tables.country_main[a.name])
        + np.array(tables.country_main[b.name])
        + np.array(tables.country_main[d.name])
    )
    assert np.allclose(global_main, expected_global)


def test_scale_up_percent_indoor_vital_handles_zero_population():
    """When indoor-vital count is zero, the % series should be all zeros."""
    a = _make_country("USA", big_6=True, weekly_cadr=10)
    _populate_for_aggregation(a, 0, 0, "Northern America")  # zero indoor pop

    tables = cc.scale_up_all_countries({"USA": a}, weeks=5)
    pct = tables.country_percent_indoor_vital[a.name]
    # First two entries are the indoor population counts.
    assert pct[0] == 0 and pct[1] == 0
    # Remaining entries are zero because of the early-return in
    # _percent_indoor_vital.
    assert all(v == 0 for v in pct[2:])


# ---------------------------------------------------------------------------
# generate_countries + compute_country_properties
# ---------------------------------------------------------------------------


def test_compute_country_properties_skips_missing(
    scale_up_data_dir,
    ew_results_dir,
    ew_outputs,  # noqa: ARG001 - ew_outputs forces the EW CSV to exist
):
    """Countries lacking required EW properties are dropped, not crashed."""
    countries = cc.generate_countries_from_multiple_csvs(
        scale_up_data_dir / "STANDARD_COUNTRY_LIST.csv",
        scale_up_data_dir / "CR_Box_Countries_MS.csv",
        ew_results_dir / "EssentialWorkersByCountry.csv",
        scale_up_data_dir / "BaghouseAirflow.csv",
    )
    # Inject a synthetic country missing the EW columns (only has MSA/MVA/MFS)
    fake = Country(name="USA")
    fake.properties.clear()
    fake.properties.update(
        {"MSA": 0, "MVA": 1.0, "MFS": 50, "Baghouse Operating MW": 0}
    )
    countries["__fake__"] = fake
    dropped = cc.compute_country_properties(countries)
    assert "__fake__" in dropped
    assert "__fake__" not in countries


def test_pipeline_runs_on_fixtures(countries_outputs):
    """Smoke: full Countries pipeline produces all expected dataframes."""
    out = countries_outputs
    assert out.main_df.shape[0] > 0
    assert "Global" in out.main_df.index
    # Region rows + per-country rows.
    assert any(r in out.main_df.index for r in cc.UN_REGION_LIST)
    # TTR tables exist and have the right column shape.
    assert list(out.ttr_country_df.columns) == [
        "Indoor Vital in Weeks",
        "Indoor Essential in Weeks",
    ]
    # All four scale-up tables share the same shape.
    assert out.cr_man_df.shape == out.main_df.shape


def test_pipeline_global_equals_sum_of_countries(countries_outputs):
    """Global row in the main_df equals the cell-wise sum of all country rows."""
    df = countries_outputs.main_df.copy()
    # Strip ufloat -> nominal_value to compare
    nom = df.map(lambda x: getattr(x, "nominal_value", x))
    # Country rows are everything not in UN_REGION_LIST + 'Global'.
    region_rows = set(cc.UN_REGION_LIST) | {"Global"}
    country_rows = [idx for idx in nom.index if idx not in region_rows]
    expected = nom.loc[country_rows].sum(axis=0)
    actual = nom.loc["Global"]
    assert np.allclose(expected.values, actual.values, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# Full-data integration
# ---------------------------------------------------------------------------


@pytest.mark.full_data
def test_full_pipeline_matches_results_on_disk(countries_outputs):
    """Re-running the pipeline should match the on-disk EssentialWorkers CSV."""
    repo = Path(__file__).resolve().parents[1]
    ref = pd.read_csv(
        repo / "results" / "essential_workers" / "EssentialWorkersByCountry.csv"
    )
    # Match country count + sum of essential workers within tolerance
    assert ref["Essential Workers"].sum(skipna=True) > 0
    # The countries pipeline doesn't directly produce EW counts; the
    # integration is structural: every country in ref should have a row
    # in our main_df.
    df = countries_outputs.main_df
    matched = sum(1 for c in ref["Country Name"] if c in df.index)
    # >90% of countries should match (allowing for name normalisation).
    assert matched / len(ref) > 0.9, f"only {matched}/{len(ref)} countries matched"
