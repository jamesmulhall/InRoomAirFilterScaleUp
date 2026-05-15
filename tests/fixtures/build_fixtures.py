"""Distil mini fixture data from the real ``data/`` and ``results/`` folders.

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
SUBSET_ISO3 = [
    "USA", "CAN", "GBR", "DEU", "FRA",
    "JPN", "CHN", "IND", "BRA", "AUS",
    "NGA", "EGY", "ZAF", "MEX", "ARG",
    "IDN", "RUS", "TUR", "KEN",
    "NZL",
]
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
    data = repo / "data"
    out = Path(__file__).resolve().parent

    # ---- Files that are already small: copy verbatim.
    for name in (
        "ISCO-08 OpinionPollCensus.xlsx",
        "Indoors_Environmentally_Controlled_data.csv",
        "ISCO_SOC_Crosswalk.csv",
        "BaghouseAirflow.csv",
        "CR_Box_Countries_MS.csv",
    ):
        src = data / name
        dst = out / src.name
        dst.write_bytes(src.read_bytes())

    # ---- Country list: keep just the SUBSET rows.
    country_list = pd.read_csv(data / "STANDARD_COUNTRY_LIST.csv", encoding="cp1252")
    country_list = country_list[country_list["ISO-3"].isin(SUBSET_ISO3)]
    country_list.to_csv(
        out / "STANDARD_COUNTRY_LIST.csv", index=False, encoding="cp1252"
    )

    # ---- Labour force xlsx: subset by Country Name (the WB file's column 0).
    lf = pd.read_excel(data / "LFData_WB_plus.xlsx", usecols=[0, 1, 3])
    lf = lf[lf["Country Name"].isin(SUBSET_LF_COUNTRY_NAMES)]
    # Pad columns 0..3 to match the original engine read shape.
    full_lf = pd.read_excel(data / "LFData_WB_plus.xlsx")
    full_lf = full_lf[full_lf["Country Name"].isin(SUBSET_LF_COUNTRY_NAMES)]
    full_lf.to_excel(out / "LFData_WB_plus.xlsx", index=False)

    # ---- ILO ISCO-08 employment csv: subset by ref_area.label
    ilo = pd.read_csv(data / "ILO_ISCO_08_GLB.csv")
    ilo = ilo[ilo["ref_area.label"].isin(SUBSET_ILO_REF_AREAS)]
    ilo.to_csv(out / "ILO_ISCO_08_GLB.csv", index=False)

    # ---- ILO country %essential xlsx: subset by header-1 'cname' column
    ilo_pct = pd.read_excel(
        data / "ILO_country_essential_workers_pct.xlsx",
        sheet_name="Sheet1",
        header=1,
    )
    keep_cnames = SUBSET_ILO_REF_AREAS + ["United Kingdom of Great Britain and Northern Ireland"]
    ilo_pct_filt = ilo_pct[ilo_pct["cname"].isin(keep_cnames)]
    # Rebuild the workbook so the header offset (header=1) is preserved.
    # We write a single empty top row and then the data so the loader still
    # works without changes.
    with pd.ExcelWriter(out / "ILO_country_essential_workers_pct.xlsx") as writer:
        pd.DataFrame([[""] * ilo_pct.shape[1]], columns=ilo_pct.columns).to_excel(
            writer, sheet_name="Sheet1", index=False, header=False
        )
        ilo_pct_filt.to_excel(
            writer, sheet_name="Sheet1", index=False, startrow=1
        )

    print(f"Wrote {len(SUBSET_ISO3)}-country fixtures into {out}")


if __name__ == "__main__":
    main()
