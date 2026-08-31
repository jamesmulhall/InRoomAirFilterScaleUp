"""Distil mini fixture data for the essential worker pipeline from ``data/``.

Run with::

    python tests/fixtures/build_fixtures.py

The output is committed to git so the test suite can run without the
real data being present. Re-run this whenever the real source files
change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# A small but representative subset of countries chosen to span every
# UN region we care about and to include both Big_6 and non-Big_6 cases.
SUBSET_LF_COUNTRY_NAMES = [
    "United States",
    "Canada",
    "United Kingdom",
    "Germany",
    "France",
    "Japan",
    "China",
    "India",
    "Brazil",
    "Australia",
    "Nigeria",
    "Egypt, Arab Rep.",
    "South Africa",
    "Mexico",
    "Argentina",
    "Indonesia",
    "Russian Federation",
    "Turkiye",
    "Kenya",
    "New Zealand",
]
SUBSET_ILO_REF_AREAS = [
    "United States",
    "Canada",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "France",
    "Japan",
    "China",
    "India",
    "Brazil",
    "Australia",
    "Nigeria",
    "Egypt",
    "South Africa",
    "Mexico",
    "Argentina",
    "Indonesia",
    "Russian Federation",
    "Turkey",
    "Kenya",
    "New Zealand",
]


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ew_data = repo / "data" / "essential_workers"
    out_ew = Path(__file__).resolve().parent / "essential_workers"
    out_ew.mkdir(parents=True, exist_ok=True)

    for name in (
        "ISCO-08 OpinionPollCensus.xlsx",
        "Indoors_Environmentally_Controlled_data.csv",
        "Indoors_Not_Environmentally_Controlled.csv",
        "ISCO_SOC_Crosswalk.csv",
    ):
        src = ew_data / name
        dst = out_ew / src.name
        dst.write_bytes(src.read_bytes())

    full_lf = pd.read_excel(ew_data / "LFData_WB_plus.xlsx")
    full_lf = full_lf[full_lf["Country Name"].isin(SUBSET_LF_COUNTRY_NAMES)]
    full_lf.to_excel(out_ew / "LFData_WB_plus.xlsx", index=False)

    ilo = pd.read_csv(ew_data / "ILO_ISCO_08_GLB.csv")
    ilo = ilo[ilo["ref_area.label"].isin(SUBSET_ILO_REF_AREAS)]
    ilo.to_csv(out_ew / "ILO_ISCO_08_GLB.csv", index=False)

    ilo_pct = pd.read_excel(
        ew_data / "ILO_country_essential_workers_pct.xlsx",
        sheet_name="Sheet1",
        header=1,
    )
    keep_cnames = SUBSET_ILO_REF_AREAS + [
        "United Kingdom of Great Britain and Northern Ireland"
    ]
    ilo_pct_filt = ilo_pct[ilo_pct["cname"].isin(keep_cnames)]
    with pd.ExcelWriter(out_ew / "ILO_country_essential_workers_pct.xlsx") as writer:
        pd.DataFrame([[""] * ilo_pct.shape[1]], columns=ilo_pct.columns).to_excel(
            writer, sheet_name="Sheet1", index=False, header=False
        )
        ilo_pct_filt.to_excel(writer, sheet_name="Sheet1", index=False, startrow=1)

    print(f"Wrote {len(SUBSET_LF_COUNTRY_NAMES)}-country fixtures into {out_ew}")


if __name__ == "__main__":
    main()
