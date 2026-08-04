#!/usr/bin/env python3
"""Filtration coverage maps and scale-up plots (ALLFED style)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_common import (
    SCALE_UP_RESULTS,
    VISUALIZATIONS_RESULTS,
    SAVE_DPI,
    add_worker_range_band,
    apply_allfed_style,
    expand_regions_to_countries,
    get_series,
    load_country_iso_map,
    load_scale_up_table,
    nominal_value,
    plot_world_choropleth,
    week_column,
)

from countries import UN_REGION_LIST  # noqa: E402

CADRPP = 100
DEFAULT_REGIONS = [
    "Global",
    "Northern America",
    "Eastern Asia",
    "Southern Africa",
    "Southern Asia",
]
STACK_LABELS = [
    "CR Box Repurposing",
    "CR Box Stock",
    "Coal Baghouse",
    "CR Box Manufacturing",
]


def _country_coverage_at_week(
    df: pd.DataFrame,
    week: int,
    *,
    vital: bool,
) -> pd.DataFrame:
    """Per-country coverage % at a given week."""
    iso_map = load_country_iso_map()
    week_col = week_column(week)
    if week_col not in df.columns:
        raise KeyError(f"Week {week} column {week_col!r} not in scale-up table")
    rows = []
    for country in df.index:
        row = df.loc[country]
        indoor_vital = nominal_value(row.iloc[0])
        indoor_essential = nominal_value(row.iloc[1])
        if vital:
            pct = nominal_value(row[week_col])
        else:
            cadr = nominal_value(row[week_col])
            if indoor_essential <= 0:
                pct = 0.0
            else:
                pct = 100.0 * cadr / (CADRPP * indoor_essential)
        rows.append({"Country Name": country, "pct": pct})
    out = pd.DataFrame(rows)
    return out.merge(iso_map, on="Country Name", how="inner")


def _region_coverage_at_week(
    df: pd.DataFrame,
    week: int,
    *,
    vital: bool,
    regions: list[str],
) -> pd.DataFrame:
    """Per-UN-region coverage % at a given week."""
    week_col = week_column(week)
    if week_col not in df.columns:
        raise KeyError(f"Week {week} column {week_col!r} not in scale-up table")
    rows = []
    for region in regions:
        if region not in df.index:
            continue
        row = df.loc[region]
        indoor_essential = nominal_value(row.iloc[1])
        if vital:
            pct = nominal_value(row[week_col])
        else:
            cadr = nominal_value(row[week_col])
            if indoor_essential <= 0:
                pct = 0.0
            else:
                pct = 100.0 * cadr / (CADRPP * indoor_essential)
        rows.append({"Region": region, "pct": pct})
    return pd.DataFrame(rows)


def plot_coverage_maps(output_dir: Path, week: int) -> None:
    pct_df = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_PERCENT_INDOOR_VITAL_MS")
    main_df = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_output_MS")

    vital_data = _country_coverage_at_week(pct_df, week, vital=True)
    plot_world_choropleth(
        vital_data,
        iso_col="Country Code",
        value_col="pct",
        title=f"Indoor vital workers covered by filtration (week {week})",
        output_path=output_dir / f"FiltrationCoverageVitalWeek{week}.png",
        legend_label="% indoor vital covered",
        vmin=0,
        vmax=100,
    )
    print(f"Wrote {output_dir / f'FiltrationCoverageVitalWeek{week}.png'}")

    essential_data = _country_coverage_at_week(main_df, week, vital=False)
    plot_world_choropleth(
        essential_data,
        iso_col="Country Code",
        value_col="pct",
        title=f"Indoor essential workers covered by filtration (week {week})",
        output_path=output_dir / f"FiltrationCoverageEssentialWeek{week}.png",
        legend_label="% indoor essential covered",
        vmin=0,
    )
    print(f"Wrote {output_dir / f'FiltrationCoverageEssentialWeek{week}.png'}")


def plot_region_coverage_maps(output_dir: Path, week: int) -> None:
    """World maps with one colour per UN region (regional aggregates)."""
    pct_df = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_PERCENT_INDOOR_VITAL_MS")
    main_df = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_output_MS")
    region_dir = output_dir / "regions"

    vital_regions = _region_coverage_at_week(
        pct_df, week, vital=True, regions=UN_REGION_LIST
    )
    vital_data = expand_regions_to_countries(vital_regions, "Region", "pct")
    plot_world_choropleth(
        vital_data,
        iso_col="Country Code",
        value_col="pct",
        cmap="RdYlGn",
        title=f"Indoor vital workers covered by filtration by UN region (week {week})",
        output_path=region_dir / f"FiltrationCoverageVitalWeek{week}.png",
        legend_label="% indoor vital covered",
        vmin=0,
        # vmax=50,
        alpha=0.8,
        hide_internal_borders=False,
    )
    print(f"Wrote {region_dir / f'FiltrationCoverageVitalWeek{week}.png'}")

    essential_regions = _region_coverage_at_week(
        main_df, week, vital=False, regions=UN_REGION_LIST
    )
    essential_data = expand_regions_to_countries(essential_regions, "Region", "pct")
    plot_world_choropleth(
        essential_data,
        iso_col="Country Code",
        value_col="pct",
        cmap="magma",
        title=(
            f"Indoor essential workers covered by filtration by UN region (week {week})"
        ),
        output_path=region_dir / f"FiltrationCoverageEssentialWeek{week}.png",
        legend_label="% indoor essential covered",
        vmin=0,
        # vmax=50,
        alpha=0.8,
        hide_internal_borders=False,
    )
    print(f"Wrote {region_dir / f'FiltrationCoverageEssentialWeek{week}.png'}")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scale_up_line(
    df_main: pd.DataFrame,
    region: str,
    output_path: Path,
    *,
    sigma: float = 1.0,
    show_start: bool = False,
) -> None:
    """Total CADR line plot with uncertainty band and vital–essential range."""
    t, vals, errs = get_series(df_main, region)
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(t, vals, label="Total CADR", linewidth=2)
    color = line.get_color()
    ax.fill_between(
        t,
        vals - sigma * errs,
        vals + sigma * errs,
        color=color,
        alpha=0.25,
        label=f"±{sigma:g}σ uncertainty",
    )
    add_worker_range_band(ax, df_main, region, alpha=0.12, color=color)
    if show_start:
        ax.set_xlim(0, 20)
    ax.set_ylabel("Clean Air Delivery Rate (L/s)")
    ax.set_xlabel("Weeks")
    ax.set_title(f"{region} scale-up with ±{sigma:g}σ bands", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_stacked_region(
    df_repur: pd.DataFrame,
    df_stock: pd.DataFrame,
    df_coalbag: pd.DataFrame,
    df_man: pd.DataFrame,
    region: str,
    output_path: Path,
    *,
    show_start: bool = False,
) -> None:
    """Stacked area CADR contributions with vital–essential worker band."""
    _, vals1, _ = get_series(df_repur, region)
    _, vals2, _ = get_series(df_stock, region)
    _, vals3, _ = get_series(df_coalbag, region)
    _, vals4, _ = get_series(df_man, region)
    t = np.arange(len(vals1))

    fig, ax = plt.subplots(figsize=(10, 6))
    add_worker_range_band(ax, df_repur, region)

    ax.stackplot(
        t,
        vals1,
        vals2,
        vals3,
        vals4,
        labels=STACK_LABELS,
        alpha=0.6,
    )

    if show_start:
        ax.set_xlim(0, 20)
        total_at_20 = vals1[20] + vals2[20] + vals3[20] + vals4[20]
        ax.set_ylim(0, total_at_20 * 1.05)

    ax.set_ylabel("Clean Air Delivery Rate (L/s)")
    ax.set_xlabel("Weeks")
    ax.set_title(f"{region} — stacked filtration scale-up", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_stacked_lines_region(
    df_repur: pd.DataFrame,
    df_stock: pd.DataFrame,
    df_coalbag: pd.DataFrame,
    df_man: pd.DataFrame,
    region: str,
    output_path: Path,
    *,
    show_start: bool = False,
) -> None:
    """Stacked line plot: cumulative CADR by source with vital–essential band."""
    _, vals1, _ = get_series(df_repur, region)
    _, vals2, _ = get_series(df_stock, region)
    _, vals3, _ = get_series(df_coalbag, region)
    _, vals4, _ = get_series(df_man, region)
    t = np.arange(len(vals1))

    cum1 = vals1
    cum2 = vals1 + vals2
    cum3 = vals1 + vals2 + vals3
    cum4 = vals1 + vals2 + vals3 + vals4
    cumulative = [cum1, cum2, cum3, cum4]

    fig, ax = plt.subplots(figsize=(10, 6))
    add_worker_range_band(ax, df_repur, region)

    for label, y in zip(STACK_LABELS, cumulative):
        ax.plot(t, y, linewidth=2, label=label)

    if show_start:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, cum4[20] * 1.05)

    ax.set_ylabel("Clean Air Delivery Rate (L/s)")
    ax.set_xlabel("Weeks")
    ax.set_title(f"{region} — stacked line filtration scale-up", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_scale_up_regions(
    output_dir: Path,
    regions: list[str],
    *,
    show_start: bool,
) -> None:
    df_main = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_output_MS")
    df_repur = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_CR_REPUR_MS")
    df_stock = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_CR_STOCK")
    df_coalbag = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_COALBAG_MS")
    df_man = load_scale_up_table(SCALE_UP_RESULTS / "Scale_up_CR_MAN_MS")

    line_dir = output_dir / "scale_up_lines"
    stacked_dir = output_dir / "stacked_scale_up"
    stacked_lines_dir = output_dir / "stacked_lines_scale_up"

    for region in regions:
        if region not in df_main.index:
            print(f"Skipping unknown region: {region}")
            continue
        slug = region.replace(" ", "_")

        plot_scale_up_line(
            df_main,
            region,
            line_dir / f"{slug}_cadr_line.png",
            show_start=False,
        )
        print(f"Wrote {line_dir / f'{slug}_cadr_line.png'}")

        plot_stacked_region(
            df_repur,
            df_stock,
            df_coalbag,
            df_man,
            region,
            stacked_dir / f"{slug}_stacked_cadr.png",
            show_start=False,
        )
        print(f"Wrote {stacked_dir / f'{slug}_stacked_cadr.png'}")

        plot_stacked_lines_region(
            df_repur,
            df_stock,
            df_coalbag,
            df_man,
            region,
            stacked_lines_dir / f"{slug}_stacked_lines_cadr.png",
            show_start=False,
        )
        print(f"Wrote {stacked_lines_dir / f'{slug}_stacked_lines_cadr.png'}")

        if show_start:
            plot_scale_up_line(
                df_main,
                region,
                line_dir / f"{slug}_cadr_line_first_20_weeks.png",
                show_start=True,
            )
            print(f"Wrote {line_dir / f'{slug}_cadr_line_first_20_weeks.png'}")
            plot_stacked_region(
                df_repur,
                df_stock,
                df_coalbag,
                df_man,
                region,
                stacked_dir / f"{slug}_stacked_cadr_first_20_weeks.png",
                show_start=True,
            )
            print(f"Wrote {stacked_dir / f'{slug}_stacked_cadr_first_20_weeks.png'}")
            plot_stacked_lines_region(
                df_repur,
                df_stock,
                df_coalbag,
                df_man,
                region,
                stacked_lines_dir / f"{slug}_stacked_lines_cadr_first_20_weeks.png",
                show_start=True,
            )
            print(
                f"Wrote {stacked_lines_dir / f'{slug}_stacked_lines_cadr_first_20_weeks.png'}"
            )


def main(
    output_dir: Path,
    week: int,
    regions: list[str],
    show_start: bool,
) -> None:
    apply_allfed_style()
    plot_coverage_maps(output_dir, week)
    plot_region_coverage_maps(output_dir, week)
    plot_scale_up_regions(output_dir, regions, show_start=show_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATIONS_RESULTS,
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=12,
        help="Week index for coverage maps (12 ≈ 3 months)",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="Regions for scale-up line and stacked plots",
    )
    parser.add_argument(
        "--show-start",
        action="store_true",
        help="Also save first-20-weeks variants",
    )
    args = parser.parse_args()
    main(args.output_dir, args.week, args.regions, args.show_start)
