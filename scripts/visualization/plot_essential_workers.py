#!/usr/bin/env python3
"""World maps of essential / vital worker shares (% of labour force)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    VISUALIZATIONS_RESULTS,
    apply_allfed_style,
    plot_world_choropleth,
)

MAP_SPECS = [
    (
        "PctEssentialWorkersByCountry.png",
        "%Essential Workers",
        "Total essential workers (% of labour force)",
    ),
    (
        "PctVitalWorkersByCountry.png",
        "%Vital Workers",
        "Total vital workers (% of labour force)",
    ),
    (
        "PctIndoorEssentialWorkersByCountry.png",
        "%Indoor Essential Workers",
        "Indoor essential workers (% of labour force)",
    ),
    (
        "PctIndoorVitalWorkersByCountry.png",
        "%Indoor Vital Workers",
        "Indoor vital workers (% of labour force)",
    ),
]


def main(output_dir: Path) -> None:
    apply_allfed_style()
    ew_path = ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv"
    df = pd.read_csv(ew_path)
    df = df.dropna(subset=["Country Code"])
    for filename, col, title in MAP_SPECS:
        plot_df = df[["Country Code", col]].copy()
        plot_df["pct"] = plot_df[col] * 100.0
        plot_df = plot_df.dropna(subset=["pct"])
        plot_world_choropleth(
            plot_df,
            iso_col="Country Code",
            value_col="pct",
            title=title,
            output_path=output_dir / filename,
            legend_label="% of labour force",
            vmin=0,
            vmax=75,
        )
        print(f"Wrote {output_dir / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATIONS_RESULTS,
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()
    main(args.output_dir)
