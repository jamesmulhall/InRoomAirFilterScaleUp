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
from preprocessing import US_ARMED_FORCES_EMPLOYMENT

# ---------------------------------------------------------------------------
# Indoor context
# ---------------------------------------------------------------------------


def test_onet_max_takes_higher_context():
    controlled = pd.DataFrame(
        {"Code": ["11-1111.00", "22-2222.00"], "Context": [50.0, 80.0]}
    )
    not_controlled = pd.DataFrame(
        {"Code": ["11-1111.00", "22-2222.00"], "Context": [90.0, 40.0]}
    )
    merged = ew._merge_onet_max_context(controlled, not_controlled)
    assert merged.loc[merged["Code"] == "11-1111", "context_pct"].iloc[0] == 90.0
    assert merged.loc[merged["Code"] == "22-2222", "context_pct"].iloc[0] == 80.0


def test_onet_banded_thresholds():
    assert ew._pct_to_indoor_fraction(80, "onet_banded") == 1.0
    assert ew._pct_to_indoor_fraction(60, "onet_banded") == 0.5
    assert ew._pct_to_indoor_fraction(40, "onet_banded") == 0.0
    assert ew._pct_to_indoor_fraction(80, "onet_max") == pytest.approx(0.8)


def test_jem_partial_buckets():
    assert ew._location_to_indoor_fraction(2.9) == 1.0
    assert ew._location_to_indoor_fraction(2.1) == 0.5
    assert ew._location_to_indoor_fraction(1.0) == 0.0
    assert ew._location_to_indoor_fraction(0.2) == 0.0


def test_jem_binary_buckets():
    assert ew._location_to_indoor_fraction(2.9, partial=False) == 1.0
    assert ew._location_to_indoor_fraction(2.1, partial=False) == 1.0
    assert ew._location_to_indoor_fraction(1.0, partial=False) == 0.0
    assert ew._location_to_indoor_fraction(0.2, partial=False) == 0.0


def test_load_indoor_context_method_reads_settings():
    assert ew.load_indoor_context_method() == "jem_binary"


def test_indoor_method_does_not_change_total_weights(data_dir):
    w_max = _weights(data_dir, indoor_context_method="onet_max")
    w_band = _weights(data_dir, indoor_context_method="onet_banded")
    pd.testing.assert_series_equal(
        w_max["ISCO_08_PollWeights_Total"],
        w_band["ISCO_08_PollWeights_Total"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        w_max["ISCO_08_ILOWeights_Total"],
        w_band["ISCO_08_ILOWeights_Total"],
        check_names=False,
    )


@pytest.mark.full_data
def test_jem_load_produces_l2_indoors_context(data_dir):
    poll = pd.read_excel(data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl")
    cw = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    w = ew.build_isco_lvl2_weights(
        poll,
        cw,
        indoor_context_method="jem_partial",
        jem_path=data_dir / "job_exposure_matrix.xls",
    )
    assert ew.INDOORS_CONTEXT_COLUMN in w.columns
    assert w[ew.INDOORS_CONTEXT_COLUMN].between(0, 1).all()


@pytest.mark.full_data
def test_compare_indoor_context_methods_returns_all_methods(data_dir):
    out = ew.compare_indoor_context_methods(data_dir)
    assert set(out["indoor_context_method"]) == set(ew.INDOOR_CONTEXT_METHODS)
    assert len(out) == len(ew.INDOOR_CONTEXT_METHODS) * 6


# ---------------------------------------------------------------------------
# Weight builder
# ---------------------------------------------------------------------------


def _weights(
    data_dir: Path,
    aggregator: str = "mean",
    indoor_context_method: ew.IndoorContextMethod = "onet_max",
) -> pd.DataFrame:
    poll = pd.read_excel(data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl")
    onet_env = pd.read_csv(data_dir / "Indoors_Environmentally_Controlled_data.csv")
    onet_not = pd.read_csv(data_dir / "Indoors_Not_Environmentally_Controlled.csv")
    cw = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    jem = data_dir / "job_exposure_matrix.xls"
    return ew.build_isco_lvl2_weights(
        poll,
        cw,
        onet_controlled_df=onet_env,
        onet_not_controlled_df=onet_not,
        indoor_context_method=indoor_context_method,
        jem_path=jem if jem.exists() else None,
        soc_to_isco_aggregator=aggregator,
    )


def test_subsistence_farmers_have_zero_indoor_context(data_dir):
    w = _weights(data_dir)
    assert w.at["63", ew.INDOORS_CONTEXT_COLUMN] == 0


def test_farming_carry_over_61_equals_62(data_dir):
    w = _weights(data_dir)
    # Code 61 should pick up code 62's Census/Context (the L4 inputs for 61
    # are missing in the ONET data, so the carry-over fills the gap).
    assert w.at["61", "Vital Weight POLL"] == w.at["62", "Vital Weight POLL"]
    assert (
        w.at["61", ew.INDOORS_CONTEXT_COLUMN] == w.at["62", ew.INDOORS_CONTEXT_COLUMN]
    )


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
    """Switching SOC->ISCO aggregator changes indoors_context but not totals.

    ``ISCO_08_*Weights_Total`` are independent of indoors_context, so they
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
    assert not w_mean[ew.INDOORS_CONTEXT_COLUMN].equals(
        w_last[ew.INDOORS_CONTEXT_COLUMN]
    )


def test_invalid_aggregator_raises(data_dir):
    poll = pd.read_excel(data_dir / "ISCO-08 OpinionPollCensus.xlsx", engine="openpyxl")
    onet_env = pd.read_csv(data_dir / "Indoors_Environmentally_Controlled_data.csv")
    onet_not = pd.read_csv(data_dir / "Indoors_Not_Environmentally_Controlled.csv")
    cw = pd.read_csv(data_dir / "ISCO_SOC_Crosswalk.csv")
    with pytest.raises(ValueError):
        ew.build_isco_lvl2_weights(
            poll,
            cw,
            onet_controlled_df=onet_env,
            onet_not_controlled_df=onet_not,
            soc_to_isco_aggregator="bogus",
        )


# ---------------------------------------------------------------------------
# Employment-by-ISCO dict
# ---------------------------------------------------------------------------


def _ilo_row(country, classif, obs, year, sex="Total"):
    return {
        "ref_area.label": country,
        "source.label": "LFS",
        "indicator.label": "Employment",
        "sex.label": sex,
        "classif1.label": classif,
        "time": year,
        "obs_value": obs,
    }


def test_build_employment_picks_latest_non_nan_year():
    """Country-year snapshot uses latest year when NEC rule does not apply (no Tot)."""
    df = pd.DataFrame(
        [
            _ilo_row(
                "Australia",
                "Occupation (ISCO-08), 2 digit level: 22 - Health",
                100.0,
                2018,
            ),
            _ilo_row(
                "Australia",
                "Occupation (ISCO-08), 2 digit level: 22 - Health",
                200.0,
                2021,
            ),
            _ilo_row(
                "Australia",
                "Occupation (ISCO-08), 2 digit level: 22 - Health",
                float("nan"),
                2024,
            ),
        ]
    )
    out = ew.build_employment_by_isco(df)
    aus = next(iter(out.values()))
    assert aus["22"] == 200_000.0


def test_build_employment_ignores_non_total_sex():
    """Male/Female rows are dropped; Total row is kept."""
    df = pd.DataFrame(
        [
            _ilo_row(
                "Barbados",
                "Occupation (ISCO-08), 2 digit level: Total",
                100.0,
                2024,
                sex="Male",
            ),
            _ilo_row(
                "Barbados",
                "Occupation (ISCO-08), 2 digit level: Total",
                500.0,
                2024,
                sex="Total",
            ),
            _ilo_row(
                "Barbados",
                "Occupation (ISCO-08), 2 digit level: 22 - Health",
                999.0,
                2024,
                sex="Female",
            ),
            _ilo_row(
                "Barbados",
                "Occupation (ISCO-08), 2 digit level: 22 - Health",
                50.0,
                2024,
                sex="Total",
            ),
        ]
    )
    out = ew.build_employment_by_isco(df)
    t = out["Barbados"]
    assert t["Tot"] == 500_000.0
    assert t["22"] == 50_000.0


def test_build_employment_year_fallback_high_nec():
    """Prefer latest year with NEC/Tot <= 10% over a more recent high-NEC year."""
    tot = "Occupation (ISCO-08), 2 digit level: Total"
    nec = "Occupation (ISCO-08), 2 digit level: Not elsewhere classified"
    health = "Occupation (ISCO-08), 2 digit level: 22 - Health"
    df = pd.DataFrame(
        [
            _ilo_row("Belize", tot, 1000.0, 2024),
            _ilo_row("Belize", nec, 800.0, 2024),
            _ilo_row("Belize", health, 50.0, 2024),
            _ilo_row("Belize", tot, 1000.0, 2023),
            _ilo_row("Belize", nec, 50.0, 2023),
            _ilo_row("Belize", health, 100.0, 2023),
        ]
    )
    out = ew.build_employment_by_isco(df)
    n = out["Belize"]
    assert n["Tot"] == 1_000_000.0
    assert n["Not"] == 50_000.0
    assert n["22"] == 100_000.0


def test_build_employment_year_fallback_all_high_nec():
    """When every year has NEC > 10%, use the year with the lowest NEC %."""
    tot = "Occupation (ISCO-08), 2 digit level: Total"
    nec = "Occupation (ISCO-08), 2 digit level: Not elsewhere classified"
    df = pd.DataFrame(
        [
            _ilo_row("Benin", tot, 100.0, 2022),
            _ilo_row("Benin", nec, 50.0, 2022),
            _ilo_row("Benin", tot, 100.0, 2023),
            _ilo_row("Benin", nec, 40.0, 2023),
            _ilo_row("Benin", tot, 100.0, 2024),
            _ilo_row("Benin", nec, 30.0, 2024),
        ]
    )
    out = ew.build_employment_by_isco(df)
    assert out["Benin"]["Not"] == 30_000.0


def test_build_employment_imputes_missing_armed_forces_security_and_laos_cleaning():
    """Missing AF and security are imputed globally; cleaning only for Laos."""
    tot = "Occupation (ISCO-08), 2 digit level: Total"
    c01 = "Occupation (ISCO-08), 2 digit level: 01 - Commissioned armed forces officers"
    c02 = "Occupation (ISCO-08), 2 digit level: 02 - Non-commissioned armed forces officers"
    c03 = "Occupation (ISCO-08), 2 digit level: 03 - Armed forces occupations, other ranks"
    c54 = "Occupation (ISCO-08), 2 digit level: 54 - Protective services workers"
    c91 = "Occupation (ISCO-08), 2 digit level: 91 - Cleaners and helpers"
    c96 = "Occupation (ISCO-08), 2 digit level: 96 - Refuse workers and other elementary workers"
    health = "Occupation (ISCO-08), 2 digit level: 22 - Health professionals"
    df = pd.DataFrame(
        [
            _ilo_row("Argentina", tot, 1000.0, 2024),
            _ilo_row("Argentina", c01, 10.0, 2024),
            _ilo_row("Argentina", c02, 20.0, 2024),
            _ilo_row("Argentina", c03, 30.0, 2024),
            _ilo_row("Argentina", c54, 54.0, 2024),
            _ilo_row("Argentina", c91, 91.0, 2024),
            _ilo_row("Argentina", c96, 96.0, 2024),
            _ilo_row("Argentina", health, 100.0, 2024),
            _ilo_row("Belize", tot, 500.0, 2024),
            _ilo_row("Belize", health, 50.0, 2024),
            _ilo_row("Lao People's Democratic Republic", tot, 200.0, 2024),
            _ilo_row("Lao People's Democratic Republic", health, 20.0, 2024),
            _ilo_row("Benin", tot, 300.0, 2024),
            _ilo_row("Benin", health, 30.0, 2024),
        ]
    )
    out = ew.build_employment_by_isco(df)
    belize_coded = 50_000.0
    laos_coded = 20_000.0
    share_91 = 91 / 401
    share_96 = 96 / 401
    share_01 = 10 / 401
    share_54 = 54 / 401

    assert out["Belize"]["01"] == pytest.approx(belize_coded * share_01)
    assert out["Belize"]["54"] == pytest.approx(belize_coded * share_54)
    assert out["Laos"]["91"] == pytest.approx(laos_coded * share_91)
    assert out["Laos"]["96"] == pytest.approx(laos_coded * share_96)
    assert "91" not in out["Benin"]
    assert "96" not in out["Benin"]


def test_build_employment_us_armed_forces_uses_dod_override():
    """US armed forces use DOD active-duty end strength, not median imputation."""
    tot = "Occupation (ISCO-08), 2 digit level: Total"
    health = "Occupation (ISCO-08), 2 digit level: 22 - Health professionals"
    df = pd.DataFrame(
        [
            _ilo_row("United States of America", tot, 150_000.0, 2024),
            _ilo_row("United States of America", health, 50_000.0, 2024),
        ]
    )
    out = ew.build_employment_by_isco(df)
    for code, expected in US_ARMED_FORCES_EMPLOYMENT.items():
        assert out["United States"][code] == expected


def test_compute_worker_dicts_nec_imputation():
    """NEC employment is counted at the coded occupations' average essential/vital weights."""
    weights = ew.build_isco_lvl2_weights(
        pd.read_excel(
            Path(__file__).resolve().parents[1]
            / "data/essential_workers/ISCO-08 OpinionPollCensus.xlsx"
        ),
        pd.read_csv(
            Path(__file__).resolve().parents[1]
            / "data/essential_workers/ISCO_SOC_Crosswalk.csv"
        ),
        pd.read_csv(
            Path(__file__).resolve().parents[1]
            / "data/essential_workers/Indoors_Environmentally_Controlled_data.csv"
        ),
        pd.read_csv(
            Path(__file__).resolve().parents[1]
            / "data/essential_workers/Indoors_Not_Environmentally_Controlled.csv"
        ),
    )
    emp = {
        "Barbados": {
            "Tot": 1000.0,
            "Not": 200.0,
            "52": 800.0,
        }
    }
    workers = ew.compute_worker_dicts(emp, weights)
    w = ew.apply_group_overlaps(weights, ew.GROUP_OVERLAP)
    wt_ilo = w.loc["52", "ISCO_08_ILOWeights_Total"]
    wt_poll = w.loc["52", "ISCO_08_PollWeights_Total"]
    coded = 800.0
    ew_coded = coded * wt_ilo
    vw_coded = coded * wt_poll
    expected_ew = ew_coded + 200.0 * (ew_coded / coded)
    expected_vw = vw_coded + 200.0 * (vw_coded / coded)
    assert workers.ew_ilo["Barbados"] == pytest.approx(expected_ew, rel=1e-9)
    assert workers.vw_poll["Barbados"] == pytest.approx(expected_vw, rel=1e-9)


def test_armed_forces_subtotal_matches_codes_01_02_03(ew_outputs):
    """Armed-forces sub-sums must equal sum of ISCO L2 01/02/03 contributions."""
    workers = ew_outputs.workers
    weights = ew_outputs.weights_df
    emp = ew_outputs.employment_by_iso
    ilo_w = weights["ISCO_08_ILOWeights"].to_dict()

    # Pick a country with armed-forces employment
    target = None
    for c, vals in emp.items():
        if any(code in vals and pd.notna(vals[code]) for code in ("01", "02", "03")):
            target = c
            break
    if target is None:
        pytest.skip("no country with armed-forces employment in this dataset")

    expected = 0.0
    for code in ew.ARMED_FORCES_L2:
        if code in emp[target] and pd.notna(emp[target][code]):
            w = ilo_w.get(code, 0)
            expected += emp[target][code] * (w if pd.notna(w) else 0)
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


def test_attach_onsite_excluded_pct_sums_61_and_63():
    lf = pd.DataFrame(
        {"Country Name": ["Nigeria", "China"], "Country Code": ["NGA", "CHN"]}
    )
    emp = {
        "Nigeria": {"Tot": 100.0, "61": 10.0, "63": 30.0},
    }
    weights = pd.DataFrame(
        {
            "ISCO_08_ILOWeights_Total": {"61": 0.8, "63": 1.0},
            "ISCO_08_PollWeights_Total": {"61": 0.5, "63": 1.0},
        },
        index=["61", "63"],
    )
    out = ew.attach_onsite_excluded_pct(lf, emp, weights)
    assert out.at[0, ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] == pytest.approx(0.38)
    assert out.at[0, ew.ONSITE_EXCLUDED_VITAL_PCT_COL] == pytest.approx(0.35)
    assert pd.isna(out.at[1, ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL])
    assert pd.isna(out.at[1, ew.ONSITE_EXCLUDED_VITAL_PCT_COL])


def test_backfill_onsite_excluded_pct_from_neighbours():
    df = pd.DataFrame(
        {
            "Country Code": ["IND", "VNM", "CHN"],
            ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL: [0.03, 0.05, np.nan],
            "%Essential Workers": [0.5, 0.5, 0.5],
            "%Indoor Essential Workers": [0.1, 0.1, 0.1],
            "%Vital Workers": [0.4, 0.4, 0.4],
            "%Indoor Vital Workers": [0.1, 0.1, 0.1],
        }
    )
    out = ew.backfill_neighbours(
        df,
        similar_iso3={"CHN": ["IND", "VNM"]},
        cols=[ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL],
    )
    assert out.at[2, ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] == pytest.approx(0.04)


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


def test_all_worker_counts_non_negative(ew_outputs):
    """Every pipeline worker headcount column must be >= 0 where present."""
    worker_count_columns = ("Labour Force (2024)",) + tuple(
        c for c, _ in ew._COUNT_COLUMNS
    )
    violations: list[str] = []

    def check(
        label: str, df: pd.DataFrame, columns: tuple[str, ...], name_col: str
    ) -> None:
        for col in columns:
            if col not in df.columns:
                continue
            bad = df.loc[df[col].notna() & (df[col] < 0)]
            for _, row in bad.iterrows():
                name = row[name_col] if name_col in row.index else row.name
                violations.append(f"{label} {name} {col}={row[col]:.4g}")

    check("country", ew_outputs.labour_force_df, worker_count_columns, "Country Name")
    check("region", ew_outputs.regional_df, worker_count_columns, "Region")
    check(
        "onsite_housing",
        ew_outputs.onsite_housing_df,
        ew.ONSITE_HOUSING_WORKER_COUNT_COLUMNS,
        "Country Name",
    )

    summary = ew.compute_global_worker_summary(ew_outputs.labour_force_df)
    bad_summary = summary.loc[summary["Workers"].notna() & (summary["Workers"] < 0)]
    for idx, row in bad_summary.iterrows():
        violations.append(f"global_summary {idx} Workers={row['Workers']:.4g}")

    assert not violations, "Negative worker counts:\n" + "\n".join(violations)


def test_onsite_housing_excludes_isco_61_and_63(ew_outputs):
    """Housing: essential/vital subtract ILO- and poll-weighted 61+63 shares."""
    housing = ew_outputs.onsite_housing_df
    countries = housing.loc[housing["Country Code"] != "GLOBAL"]
    global_row = housing.loc[housing["Country Code"] == "GLOBAL"].iloc[0]

    assert "Essential Workers (Housing Requirement)" in housing.columns
    assert "Vital Workers (Housing Requirement)" in housing.columns
    assert global_row["Essential Workers (Housing Requirement)"] == pytest.approx(
        countries["Essential Workers (Housing Requirement)"].sum()
    )
    assert global_row["Vital Workers (Housing Requirement)"] == pytest.approx(
        countries["Vital Workers (Housing Requirement)"].sum()
    )

    lf_nigeria = ew_outputs.labour_force_df.loc[
        ew_outputs.labour_force_df["Country Name"] == "Nigeria"
    ].iloc[0]
    lf = lf_nigeria["Labour Force (2024)"]
    nigeria = countries[countries["Country Name"] == "Nigeria"].iloc[0]
    expected_ess = (
        lf_nigeria["Essential Workers"]
        - lf_nigeria[ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] * lf
    )
    expected_vit = (
        lf_nigeria["Vital Workers"] - lf_nigeria[ew.ONSITE_EXCLUDED_VITAL_PCT_COL] * lf
    )
    assert nigeria["Essential Workers (Housing Requirement)"] == pytest.approx(
        expected_ess, rel=1e-6
    )
    assert nigeria["Vital Workers (Housing Requirement)"] == pytest.approx(
        expected_vit, rel=1e-6
    )
    assert lf_nigeria[ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] > 0
    assert lf_nigeria[ew.ONSITE_EXCLUDED_VITAL_PCT_COL] > 0
    assert len(housing) == len(ew_outputs.labour_force_df) + 1


@pytest.mark.full_data
def test_china_onsite_excluded_pct_backfilled_from_neighbours(ew_outputs):
    """Countries without ILO microdata get ISCO 61+63 share from SIMILAR_ISO3."""
    china = ew_outputs.labour_force_df.loc[
        ew_outputs.labour_force_df["Country Code"] == "CHN"
    ].iloc[0]
    assert not pd.isna(china[ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL])
    assert china[ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL] > 0
    # neighbours JPN, IND, VNM in SIMILAR_ISO3
    for col in (ew.ONSITE_EXCLUDED_ESSENTIAL_PCT_COL, ew.ONSITE_EXCLUDED_VITAL_PCT_COL):
        assert china[col] == pytest.approx(
            (
                ew_outputs.labour_force_df.loc[
                    ew_outputs.labour_force_df["Country Code"] == "JPN", col
                ].iloc[0]
                + ew_outputs.labour_force_df.loc[
                    ew_outputs.labour_force_df["Country Code"] == "IND", col
                ].iloc[0]
                + ew_outputs.labour_force_df.loc[
                    ew_outputs.labour_force_df["Country Code"] == "VNM", col
                ].iloc[0]
            )
            / 3,
            rel=1e-6,
        )


def test_compute_global_worker_summary_outdoor_is_residual(ew_outputs):
    """Outdoor essential/vital = total minus indoor for each series."""
    summary = ew.compute_global_worker_summary(ew_outputs.labour_force_df)
    lf = ew_outputs.labour_force_df
    essential = lf["Essential Workers"].sum(skipna=True)
    indoor_essential = lf["Indoor Essential Workers"].sum(skipna=True)
    vital = lf["Vital Workers"].sum(skipna=True)
    indoor_vital = lf["Indoor Vital Workers"].sum(skipna=True)
    assert summary.loc["Outdoor essential workers", "Workers"] == pytest.approx(
        essential - indoor_essential
    )
    assert summary.loc["Outdoor vital workers", "Workers"] == pytest.approx(
        vital - indoor_vital
    )
    assert summary.loc["Essential workers", "% of Labour Force"] == pytest.approx(
        ew_outputs.validation.global_pct_essential, rel=1e-9
    )
    assert summary.loc["Vital workers", "% of Labour Force"] == pytest.approx(
        ew_outputs.validation.global_pct_vital, rel=1e-9
    )
    assert summary.loc["Essential workers", "Country min %"] == pytest.approx(
        (lf["%Essential Workers"] * 100.0).min(skipna=True)
    )
    assert summary.loc["Essential workers", "Country max %"] == pytest.approx(
        (lf["%Essential Workers"] * 100.0).max(skipna=True)
    )
    assert (
        summary.loc["Essential workers", "Country min %"]
        <= summary.loc["Essential workers", "Country max %"]
    )


def test_fill_missing_labour_force_from_ilo_tot():
    lf_df = pd.DataFrame(
        {
            "Country Name": ["Palestine", "Nigeria"],
            "Country Code": ["PSE", "NGA"],
            "Labour Force (2024)": [np.nan, 80_000_000.0],
        }
    )
    employment = {
        "Palestine": {"Tot": 719_891.0, "22": 23_046.0},
        "Nigeria": {"Tot": 71_000_000.0},
    }
    out = ew.fill_missing_labour_force_from_ilo_tot(lf_df, employment)
    assert out.loc[out["Country Code"] == "PSE", "Labour Force (2024)"].iloc[
        0
    ] == pytest.approx(719_891.0)
    assert out.loc[out["Country Code"] == "NGA", "Labour Force (2024)"].iloc[
        0
    ] == pytest.approx(80_000_000.0)


@pytest.mark.full_data
def test_palestine_worker_counts_after_ilo_tot_lf_fallback(ew_outputs):
    row = ew_outputs.labour_force_df.loc[
        ew_outputs.labour_force_df["Country Name"] == "Palestine"
    ].iloc[0]
    assert row["Labour Force (2024)"] == pytest.approx(719_891.0, rel=1e-4)
    assert row["Essential Workers"] == pytest.approx(
        row["%Essential Workers"] * row["Labour Force (2024)"], rel=1e-6
    )
    assert row["Vital Workers"] > 0
    housing = ew_outputs.onsite_housing_df.loc[
        ew_outputs.onsite_housing_df["Country Name"] == "Palestine"
    ].iloc[0]
    assert housing["Essential Workers (Housing Requirement)"] > 0


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
        "Our %Essential (model, global overlap)",
        "Our %Essential (calibrated)",
        "ILO %essential (published)",
        "ILO %essential non-agri (published)",
        "Delta model (pp)",
        "Delta calibrated (pp)",
        "Armed Forces (Essential)",
    }
    assert expected.issubset(set(out.columns))


# ---------------------------------------------------------------------------
# Group overlap calibration
# ---------------------------------------------------------------------------


def test_calibrate_group_overlaps_raise_hits_target():
    masses = {"Food": 100.0, "Manual": 50.0, "ArmedForces": 10.0}
    for g in ew.GROUP_OVERLAP:
        if g not in masses:
            masses[g] = 0.0
    e0 = ew.essential_mass_at_overlaps(masses, ew.GROUP_OVERLAP)
    target = e0 + 20.0
    overlaps, x, direction, status = ew.calibrate_group_overlaps(masses, target)
    assert direction == "raise"
    assert status in ("ok", "infeasible_clipped")
    e1 = ew.essential_mass_at_overlaps(masses, overlaps)
    assert e1 == pytest.approx(target, rel=1e-6)
    assert overlaps["ArmedForces"] == ew.ARMED_FORCES_OVERLAP_FIXED
    for g in ew.CALIBRATABLE_GROUPS:
        assert 0.0 <= overlaps[g] <= 1.0


def test_calibrate_group_overlaps_lower_hits_target():
    masses = {g: 100.0 for g in ew.GROUP_OVERLAP}
    e0 = ew.essential_mass_at_overlaps(masses, ew.GROUP_OVERLAP)
    target = e0 * 0.85
    overlaps, x, direction, _status = ew.calibrate_group_overlaps(masses, target)
    assert direction == "lower"
    e1 = ew.essential_mass_at_overlaps(masses, overlaps)
    assert e1 == pytest.approx(target, rel=1e-6)


def test_calibrated_overlaps_in_unit_interval():
    masses = {g: float(i + 1) for i, g in enumerate(ew.GROUP_OVERLAP)}
    overlaps, _x, _d, _s = ew.calibrate_group_overlaps(masses, 500.0)
    for g in ew.CALIBRATABLE_GROUPS:
        assert 0.0 <= overlaps[g] <= 1.0


def test_backfill_calibrated_overlaps_from_neighbours():
    df = pd.DataFrame(
        {
            "Country Code": ["IND", "VNM", "CHN"],
            "Country Name": ["India", "Vietnam", "China"],
            ew.overlap_column("Food"): [0.90, 0.92, np.nan],
            ew.overlap_column("Manual"): [0.40, 0.45, np.nan],
            "overlap_source": [ew.OVERLAP_SOURCE_ILO, ew.OVERLAP_SOURCE_ILO, ""],
            "calibration_x": [0.1, 0.2, np.nan],
        }
    )
    for g in ew.CALIBRATABLE_GROUPS:
        if ew.overlap_column(g) not in df.columns:
            df[ew.overlap_column(g)] = 0.5
    out = ew.backfill_calibrated_overlaps(df, similar_iso3={"CHN": ["IND", "VNM"]})
    assert out.at[2, ew.overlap_column("Food")] == pytest.approx(0.91)
    assert out.at[2, "overlap_source"] == ew.OVERLAP_SOURCE_NEIGHBOUR


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


def _feasible_ilo_validation(ew_outputs) -> pd.DataFrame:
    """Validation rows where single-knob overlap calibration reached ILO target."""
    ct = ew_outputs.overlap_calibration.country_table
    feasible = ct.loc[ew.overlap_calibration_feasible(ct), "Country Name"]
    merged = ew_outputs.validation.merged_df
    return merged[merged["Country Name"].isin(feasible)]


@pytest.mark.full_data
def test_no_country_deviates_more_than_10pp(ew_outputs):
    """Feasible ILO-calibrated countries within 10pp of published %essential."""
    subset = _feasible_ilo_validation(ew_outputs)
    bad = subset.loc[subset["Delta (pp)"].abs() > 10.0]
    assert (
        bad.empty
    ), "Feasible calibrated %Essential deviates from ILO by more than 10pp:\n" + bad[
        [
            "Country Name",
            "Our %Essential (calibrated)",
            "ILO %essential (published)",
            "Delta (pp)",
        ]
    ].to_string(
        index=False
    )


@pytest.mark.full_data
def test_model_global_overlap_can_deviate_from_ilo(ew_outputs):
    """Pre-calibration series may exceed 10pp (documents overlap cost)."""
    outliers = ew_outputs.validation_model.outlier_df
    assert not outliers.empty


@pytest.mark.full_data
def test_calibrated_essential_matches_ilo_where_employment(ew_outputs):
    """Feasible ILO-calibrated countries match published % within 0.01pp."""
    subset = _feasible_ilo_validation(ew_outputs)
    assert len(subset) > 0
    max_delta = subset["Delta (pp)"].abs().max()
    assert (
        max_delta <= ew.ESSENTIAL_PCT_TOLERANCE_PP + 1e-9
    ), f"Max feasible calibrated |Delta| is {max_delta:.4f}pp"


@pytest.mark.full_data
def test_infeasible_clipped_countries_documented(ew_outputs):
    """Countries needing x>1 are flagged (e.g. Liberia)."""
    ct = ew_outputs.overlap_calibration.country_table
    infeasible = ct.loc[ct["solver_status"] == "infeasible_clipped", "Country Name"]
    assert "Liberia" in set(infeasible)


@pytest.mark.full_data
def test_china_overlap_backfilled_from_neighbours(ew_outputs):
    """CHN without ILO employment gets mean of calibrated SIMILAR_ISO3 neighbours."""
    ct = ew_outputs.overlap_calibration.country_table
    china = ct.loc[ct["Country Code"] == "CHN"].iloc[0]
    assert china["overlap_source"] == ew.OVERLAP_SOURCE_NEIGHBOUR
    food_col = ew.overlap_column("Food")
    cal_neighbours = [
        iso
        for iso in ew.SIMILAR_ISO3["CHN"]
        if ct.loc[ct["Country Code"] == iso, "overlap_source"].iloc[0]
        in (ew.OVERLAP_SOURCE_ILO, ew.OVERLAP_SOURCE_NEIGHBOUR)
    ]
    assert "IND" in cal_neighbours
    expected = sum(
        float(ct.loc[ct["Country Code"] == iso, food_col].iloc[0])
        for iso in cal_neighbours
    ) / len(cal_neighbours)
    assert china[food_col] == pytest.approx(expected, rel=1e-6)


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


# ---------------------------------------------------------------------------
# ASHRAE-241 CADR requirements
# ---------------------------------------------------------------------------


def test_group_and_country_cadr_requirements(data_dir):
    """Two-group synthetic country: group CADR sums to country totals."""
    flat_overlap = {g: 1.0 for g in ew.GROUP_OVERLAP}
    flat_overlap["ArmedForces"] = ew.ARMED_FORCES_OVERLAP_FIXED
    template = pd.DataFrame(
        {
            "Group": {"22": "Health", "52": "Retail"},
            "Essential Weight ILO": {"22": 1, "52": 1},
            "Vital Weight POLL": {"22": 1.0, "52": 0.5},
            ew.INDOORS_CONTEXT_COLUMN: {"22": 1.0, "52": 1.0},
        }
    )
    employment = {"Testland": {"22": 600.0, "52": 400.0, "Tot": 1000.0}}
    lf_df = pd.DataFrame(
        [
            {
                "Country Name": "Testland",
                "Country Code": "TST",
                "Region": "Northern Europe",
                "Labour Force (2024)": 1_000_000.0,
                "Indoor Essential Workers": 1_000_000.0,
                "Indoor Vital Workers": 800_000.0,
            }
        ]
    )

    group_df = ew.compute_group_workers_and_cadr(
        data_dir,
        lf_df,
        employment,
        template,
        {"Testland": flat_overlap},
    )
    health = group_df.loc[group_df["occupational_group"] == "Health"].iloc[0]
    retail = group_df.loc[group_df["occupational_group"] == "Retail"].iloc[0]

    assert health["Indoor Essential Workers"] == pytest.approx(600_000)
    assert health["Indoor Vital Workers"] == pytest.approx(600_000)
    assert retail["Indoor Essential Workers"] == pytest.approx(400_000)
    assert retail["Indoor Vital Workers"] == pytest.approx(200_000)

    scaled_health = 35 * ew.ASHRAE_SCALE_FACTOR
    scaled_retail = 20 * ew.ASHRAE_SCALE_FACTOR
    assert health[ew.SCALED_ECA_COL] == pytest.approx(scaled_health)
    assert health[ew.INDOOR_ESSENTIAL_CADR_COL] == pytest.approx(
        600_000 * scaled_health
    )
    assert retail[ew.INDOOR_VITAL_CADR_COL] == pytest.approx(200_000 * scaled_retail)

    country = ew.attach_country_cadr_from_groups(lf_df, group_df)
    assert country[ew.INDOOR_ESSENTIAL_CADR_COL].iloc[0] == pytest.approx(
        health[ew.INDOOR_ESSENTIAL_CADR_COL] + retail[ew.INDOOR_ESSENTIAL_CADR_COL]
    )
    total_essential = 1_000_000.0
    expected_eca = (600_000 * scaled_health + 400_000 * scaled_retail) / total_essential
    assert country[ew.SCALED_ECA_ESSENTIAL_COL].iloc[0] == pytest.approx(expected_eca)


def test_pipeline_includes_group_and_cadr_outputs(ew_outputs):
    lf = ew_outputs.labour_force_df
    group_df = ew_outputs.group_df
    assert not group_df.empty
    assert set(ew.GROUP_OVERLAP) <= set(group_df["occupational_group"])
    for col in (
        ew.INDOOR_ESSENTIAL_CADR_COL,
        ew.INDOOR_VITAL_CADR_COL,
        ew.SCALED_ECA_ESSENTIAL_COL,
        ew.SCALED_ECA_VITAL_COL,
    ):
        assert col in lf.columns
    assert lf[ew.INDOOR_ESSENTIAL_CADR_COL].notna().any()


def test_food_share_of_workforce_ratios():
    group_df = pd.DataFrame(
        [
            {
                "Country Name": "A",
                "Country Code": "AAA",
                "Region": "R",
                "occupational_group": "Food",
                "Essential Workers": 80.0,
                "Vital Workers": 70.0,
                "Indoor Essential Workers": 20.0,
                "Indoor Vital Workers": 10.0,
            },
            {
                "Country Name": "A",
                "Country Code": "AAA",
                "Region": "R",
                "occupational_group": "Health",
                "Essential Workers": 20.0,
                "Vital Workers": 30.0,
                "Indoor Essential Workers": 20.0,
                "Indoor Vital Workers": 30.0,
            },
        ]
    )
    food = ew.food_share_of_workforce(group_df)
    assert food["Food % of Essential Workers"].iloc[0] == pytest.approx(0.8)
    assert food["Food % of Vital Workers"].iloc[0] == pytest.approx(0.7)
    assert food["Food % of Indoor Essential Workers"].iloc[0] == pytest.approx(0.5)
    assert food["Food % of Indoor Vital Workers"].iloc[0] == pytest.approx(0.25)


def test_summarize_group_composition_global_and_country():
    group_df = pd.DataFrame(
        [
            {
                "Country Name": "A",
                "Country Code": "AAA",
                "Region": "Northern Europe",
                "occupational_group": "Food",
                "Essential Workers": 80.0,
                "Vital Workers": 40.0,
                "Indoor Essential Workers": 10.0,
                "Indoor Vital Workers": 5.0,
            },
            {
                "Country Name": "A",
                "Country Code": "AAA",
                "Region": "Northern Europe",
                "occupational_group": "Health",
                "Essential Workers": 20.0,
                "Vital Workers": 60.0,
                "Indoor Essential Workers": 30.0,
                "Indoor Vital Workers": 45.0,
            },
            {
                "Country Name": "B",
                "Country Code": "BBB",
                "Region": "Northern Europe",
                "occupational_group": "Food",
                "Essential Workers": 20.0,
                "Vital Workers": 10.0,
                "Indoor Essential Workers": 5.0,
                "Indoor Vital Workers": 2.0,
            },
            {
                "Country Name": "B",
                "Country Code": "BBB",
                "Region": "Northern Europe",
                "occupational_group": "Health",
                "Essential Workers": 80.0,
                "Vital Workers": 90.0,
                "Indoor Essential Workers": 70.0,
                "Indoor Vital Workers": 80.0,
            },
        ]
    )
    global_comp = ew.summarize_group_composition(group_df)
    food = global_comp.loc[global_comp["occupational_group"] == "Food"].iloc[0]
    # Worker-weighted: Food essential = (80+20)/(100+100) = 0.5
    assert food["% of Essential Workers"] == pytest.approx(0.5)
    assert food["% of Vital Workers"] == pytest.approx(0.25)
    assert global_comp["% of Essential Workers"].sum() == pytest.approx(1.0)

    country = ew.summarize_group_composition(group_df, by="Country")
    a_food = country.loc[
        (country["Country Code"] == "AAA") & (country["occupational_group"] == "Food")
    ].iloc[0]
    assert a_food["% of Essential Workers"] == pytest.approx(0.8)

    region = ew.summarize_group_composition(group_df, by="Region")
    assert region["% of Essential Workers"].sum() == pytest.approx(1.0)


def test_summarize_worker_shares_vs_gdp_negative_association():
    # Higher GDP → lower essential share (synthetic).
    lf_df = pd.DataFrame(
        {
            "Country Name": ["Poor", "Mid", "Rich"],
            "Country Code": ["POO", "MID", "RIC"],
            "Region": ["R", "R", "R"],
            "%Essential Workers": [0.7, 0.5, 0.3],
            "%Indoor Essential Workers": [0.25, 0.22, 0.20],
            "%Vital Workers": [0.6, 0.4, 0.2],
            "%Indoor Vital Workers": [0.18, 0.16, 0.14],
        }
    )
    group_df = pd.DataFrame(
        [
            {
                "Country Name": "Poor",
                "Country Code": "POO",
                "Region": "R",
                "occupational_group": "Food",
                "Essential Workers": 90.0,
                "Vital Workers": 90.0,
                "Indoor Essential Workers": 10.0,
                "Indoor Vital Workers": 10.0,
            },
            {
                "Country Name": "Poor",
                "Country Code": "POO",
                "Region": "R",
                "occupational_group": "Health",
                "Essential Workers": 10.0,
                "Vital Workers": 10.0,
                "Indoor Essential Workers": 10.0,
                "Indoor Vital Workers": 10.0,
            },
            {
                "Country Name": "Rich",
                "Country Code": "RIC",
                "Region": "R",
                "occupational_group": "Food",
                "Essential Workers": 10.0,
                "Vital Workers": 10.0,
                "Indoor Essential Workers": 5.0,
                "Indoor Vital Workers": 5.0,
            },
            {
                "Country Name": "Rich",
                "Country Code": "RIC",
                "Region": "R",
                "occupational_group": "Health",
                "Essential Workers": 90.0,
                "Vital Workers": 90.0,
                "Indoor Essential Workers": 90.0,
                "Indoor Vital Workers": 90.0,
            },
            {
                "Country Name": "Mid",
                "Country Code": "MID",
                "Region": "R",
                "occupational_group": "Food",
                "Essential Workers": 50.0,
                "Vital Workers": 50.0,
                "Indoor Essential Workers": 20.0,
                "Indoor Vital Workers": 20.0,
            },
            {
                "Country Name": "Mid",
                "Country Code": "MID",
                "Region": "R",
                "occupational_group": "Health",
                "Essential Workers": 50.0,
                "Vital Workers": 50.0,
                "Indoor Essential Workers": 50.0,
                "Indoor Vital Workers": 50.0,
            },
        ]
    )
    gdp_df = pd.DataFrame(
        {
            "Country Code": ["POO", "MID", "RIC"],
            ew.GDP_PPP_COL: [2000.0, 10000.0, 50000.0],
        }
    )
    _, summary = ew.summarize_worker_shares_vs_gdp(lf_df, group_df, gdp_df)
    ess = summary.loc[summary["column"] == "%Essential Workers"].iloc[0]
    food = summary.loc[summary["column"] == "Food % of Essential Workers"].iloc[0]
    assert ess["Spearman ρ"] < 0
    assert food["Spearman ρ"] < 0
