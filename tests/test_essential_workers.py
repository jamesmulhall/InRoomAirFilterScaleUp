"""Tests for the essential-worker pipeline in ``src/essential_workers.py``.

Run the fast subset with::

    pytest

Run including the slow ILO sense-checks against the real data with::

    pytest --full-data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import essential_workers as ew


# ---------------------------------------------------------------------------
# Weight builder
# ---------------------------------------------------------------------------


def _weights(data_dir: Path, aggregator: str = "mean") -> pd.DataFrame:
    poll = pd.read_excel(data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl")
    onet = pd.read_csv(data_dir / "Indoors_Environmentally_Controlled_data.csv")
    cw = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    return ew.build_isco_lvl2_weights(poll, onet, cw, soc_to_isco_aggregator=aggregator)


def test_subsistence_farmers_have_zero_indoor_context(data_dir):
    w = _weights(data_dir)
    assert w.at["63", "Context Proj"] == 0


def test_farming_carry_over_61_equals_62(data_dir):
    w = _weights(data_dir)
    # Code 61 should pick up code 62's Census/Context (the L4 inputs for 61
    # are missing in the ONET data, so the carry-over fills the gap).
    assert w.at["61", "Vital Weight POLL"] == w.at["62", "Vital Weight POLL"]
    assert w.at["61", "Context Proj"] == w.at["62", "Context Proj"]


def test_non_ilo_poll_codes_have_zero_vital_poll(data_dir):
    w = _weights(data_dir)
    for code in ew.NON_ILO_POLL_CODES:
        if code in w.index:
            assert w.at[code, "Vital Weight POLL"] == 0


def test_group_overlap_matches_constant(data_dir):
    w = _weights(data_dir)
    for code, group in ew.ISCO_L2_TO_GROUP.items():
        if code in w.index:
            assert w.at[code, "Group Overlap"] == pytest.approx(ew.GROUP_OVERLAP[group])


def test_essential_weight_ilo_is_binary(data_dir):
    w = _weights(data_dir)
    assert set(w["Essential Weight ILO"].unique()).issubset({0, 1})
    essential_codes = {f"{c:02d}" for c in ew.ILO_LVL2_ESSENTIAL_GROUPS}
    for code in essential_codes:
        if code in w.index:
            assert w.at[code, "Essential Weight ILO"] == 1


def test_aggregator_choice_affects_indoor_only(data_dir):
    """Switching SOC->ISCO aggregator changes Context Proj but not totals.

    ``ISCO_08_*Weights_Total`` are independent of Context Proj, so they
    should be identical across "mean" and "last" aggregations.
    """
    w_mean = _weights(data_dir, "mean")
    w_last = _weights(data_dir, "last")
    pd.testing.assert_series_equal(
        w_mean["ISCO_08_PollWeights_Total"],
        w_last["ISCO_08_PollWeights_Total"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        w_mean["ISCO_08_ILOWeights_Total"],
        w_last["ISCO_08_ILOWeights_Total"],
        check_names=False,
    )
    # And Context Proj must differ for at least one code (otherwise the
    # flag is doing nothing).
    assert not w_mean["Context Proj"].equals(w_last["Context Proj"])


def test_invalid_aggregator_raises(data_dir):
    poll = pd.read_excel(data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl")
    onet = pd.read_csv(data_dir / "Indoors_Environmentally_Controlled_data.csv")
    cw = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    with pytest.raises(ValueError):
        ew.build_isco_lvl2_weights(poll, onet, cw, soc_to_isco_aggregator="bogus")


# ---------------------------------------------------------------------------
# Employment-by-ISCO dict
# ---------------------------------------------------------------------------


def test_build_employment_picks_latest_non_nan_year():
    """``build_employment_by_isco`` keeps the latest non-NaN value per code."""
    df = pd.DataFrame(
        [
            {
                "ref_area.label": "Australia",
                "classif1.label": "Occupation (ISCO-08), 2 digit level: 22 - Health",
                "obs_value": 100.0,
                "time": 2018,
            },
            {
                "ref_area.label": "Australia",
                "classif1.label": "Occupation (ISCO-08), 2 digit level: 22 - Health",
                "obs_value": 200.0,
                "time": 2021,
            },
            {
                "ref_area.label": "Australia",
                "classif1.label": "Occupation (ISCO-08), 2 digit level: 22 - Health",
                "obs_value": float("nan"),
                "time": 2024,
            },
        ]
    )
    out = ew.build_employment_by_isco(df)
    aus = next(iter(out.values()))
    # employment is obs_value * 1000
    assert aus["22 "] == 200_000.0  # 2021 wins (latest non-NaN)


def test_armed_forces_subtotal_matches_codes_01_02_03(ew_outputs):
    """Armed-forces sub-sums must equal sum of ISCO L2 01/02/03 contributions."""
    workers = ew_outputs.workers
    weights = ew_outputs.weights_df
    emp = ew_outputs.employment_by_iso
    ilo_w = weights["ISCO_08_ILOWeights"].to_dict()

    # Pick a country with armed-forces employment
    target = None
    for c, vals in emp.items():
        if any(
            (f"{code} ") in vals and pd.notna(vals[f"{code} "])
            for code in ("01", "02", "03")
        ):
            target = c
            break
    if target is None:
        pytest.skip("no country with armed-forces employment in this dataset")

    expected = 0.0
    for code in ew.ARMED_FORCES_L2:
        key = f"{code} "
        if key in emp[target] and pd.notna(emp[target][key]):
            w = ilo_w.get(code, 0)
            expected += emp[target][key] * (w if pd.notna(w) else 0)
    assert workers.af_indoor_essential[target] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Labour-force back-fill and absolute counts
# ---------------------------------------------------------------------------


def test_backfill_neighbours_averages_listed_neighbours():
    """Missing country pct should be filled with mean of similar_iso3 entries."""
    df = pd.DataFrame(
        {
            "Country Code": ["AAA", "BBB", "CCC", "ZZZ"],
            "%Indoor Essential Workers": [0.20, 0.30, 0.40, np.nan],
            "%Essential Workers": [0.50, 0.60, 0.70, np.nan],
            "%Indoor Vital Workers": [0.10, 0.15, 0.20, np.nan],
            "%Vital Workers": [0.40, 0.45, 0.50, np.nan],
        }
    )
    out = ew.backfill_neighbours(df, similar_iso3={"ZZZ": ["AAA", "BBB", "CCC"]})
    assert out.at[3, "%Indoor Essential Workers"] == pytest.approx(0.30)
    assert out.at[3, "%Essential Workers"] == pytest.approx(0.60)


def test_backfill_iterates_until_stable():
    """If a neighbour itself is NaN it should still be filled in a later sweep."""
    df = pd.DataFrame(
        {
            "Country Code": ["A", "B", "C"],
            "%Indoor Essential Workers": [0.20, np.nan, np.nan],
            "%Essential Workers": [0.50, np.nan, np.nan],
            "%Indoor Vital Workers": [0.10, np.nan, np.nan],
            "%Vital Workers": [0.40, np.nan, np.nan],
        }
    )
    out = ew.backfill_neighbours(df, similar_iso3={"B": ["A"], "C": ["B"]})
    assert out.at[2, "%Indoor Essential Workers"] == pytest.approx(0.20)


def test_compute_absolute_counts_equals_pct_times_lf():
    df = pd.DataFrame(
        {
            "Country Code": ["AAA"],
            "Labour Force (2024)": [1_000_000.0],
            "%Indoor Essential Workers": [0.25],
            "%Indoor Vital Workers": [0.10],
            "%Essential Workers": [0.55],
            "%Vital Workers": [0.30],
            "%Armed Forces (Indoor Essential)": [0.01],
            "%Armed Forces (Essential)": [0.01],
        }
    )
    out = ew.compute_absolute_counts(df)
    assert out.at[0, "Indoor Essential Workers"] == pytest.approx(250_000.0)
    assert out.at[0, "Essential Workers"] == pytest.approx(550_000.0)
    assert out.at[0, "Armed Forces (Essential)"] == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Regional aggregation
# ---------------------------------------------------------------------------


def test_aggregate_by_region_sums_countries(ew_outputs):
    """For each region, region totals must equal the sum of its countries."""
    lf = ew_outputs.labour_force_df
    reg = ew_outputs.regional_df
    for region in reg["Region"].unique():
        subset = lf[lf["Region"] == region]
        for col in (
            "Labour Force (2024)",
            "Indoor Essential Workers",
            "Essential Workers",
        ):
            assert reg.loc[reg["Region"] == region, col].iloc[0] == pytest.approx(
                subset[col].sum(skipna=True), rel=1e-9
            )


def test_aggregate_by_region_pct_consistent(ew_outputs):
    """%Essential at region level == regional Essential / regional LF."""
    reg = ew_outputs.regional_df
    derived = reg["Essential Workers"] / reg["Labour Force (2024)"]
    pd.testing.assert_series_equal(
        derived, reg["%Essential Workers"], check_names=False
    )


# ---------------------------------------------------------------------------
# Validation CSV schema
# ---------------------------------------------------------------------------


def test_validation_csv_has_expected_columns(ew_outputs, results_dir):
    out = pd.read_csv(results_dir / "Essential_Workers_Validation.csv")
    expected = {
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
    }
    assert expected.issubset(set(out.columns))


# ---------------------------------------------------------------------------
# Sense checks (ILO 2023)
# ---------------------------------------------------------------------------


@pytest.mark.full_data
def test_global_essential_within_5pp_of_ilo(ew_outputs):
    """Global %Essential must be within 5pp of the ILO published 52%."""
    pct = ew_outputs.validation.global_pct_essential
    assert abs(pct - 52.0) <= 5.0, (
        f"Global %Essential is {pct:.2f}%, off the 52% ILO target by "
        f"{abs(pct-52.0):.2f}pp (> 5pp tolerance)."
    )


@pytest.mark.full_data
def test_no_country_deviates_more_than_10pp(ew_outputs):
    """Strict: no per-country |Delta| above 10pp vs ILO published figure.

    This test is expected to fail until the pipeline is improved - it
    currently flags seven outliers (Liberia, Tuvalu, Micronesia FS, Laos,
    Nigeria, Mexico, Madagascar).
    """
    outliers = ew_outputs.validation.outlier_df
    assert (
        outliers.empty
    ), "Per-country %Essential deviates from ILO by more than 10pp for " f"{len(outliers)} countries:\n" + outliers[
        [
            "Country Name",
            "Our %Essential (pct)",
            "ILO %essential (published)",
            "Delta (pp)",
        ]
    ].to_string(
        index=False
    )


@pytest.mark.full_data
def test_correlation_with_ilo_is_high(ew_outputs):
    """Per-country %Essential should correlate strongly with ILO's figure."""
    assert ew_outputs.validation.correlation >= 0.8


def test_pipeline_runs_on_subset(ew_outputs):
    """Always-on smoke test: pipeline produces a non-empty labour-force DF."""
    lf = ew_outputs.labour_force_df
    assert len(lf) > 0
    assert "%Essential Workers" in lf.columns
    # Most rows should be filled (small fixtures may leave a few NaN
    # because of sparse SIMILAR_ISO3 neighbours).
    assert lf["%Essential Workers"].notna().mean() > 0.8


@pytest.mark.full_data
def test_full_data_backfill_fills_every_country(ew_outputs):
    """On the real data the neighbour back-fill must fill every row."""
    lf = ew_outputs.labour_force_df
    for col in (
        "%Indoor Essential Workers",
        "%Indoor Vital Workers",
        "%Essential Workers",
        "%Vital Workers",
    ):
        assert lf[col].notna().all(), f"{col} still has NaN rows on full data"
