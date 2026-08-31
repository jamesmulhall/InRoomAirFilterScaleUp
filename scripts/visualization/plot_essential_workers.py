#!/usr/bin/env python3
"""World maps of essential / vital worker shares (% of labour force).

Produces three manuscript figures: a stacked vital pair, a stacked essential
pair, and one 2x2 grid holding all four panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    SAVE_DPI,
    VISUALIZATIONS_RESULTS,
    apply_allfed_style,
    draw_world_choropleth,
    label_panel,
)

VMIN = 0.0
VMAX = 75.0
CMAP = "viridis"
LEGEND_LABEL = "% of labour force"

# Manuscript panels: (panel label, column, title)
VITAL_STACK = [
    ("a", "%Vital Workers", "Total vital workers (% of labour force)"),
    ("b", "%Indoor Vital Workers", "Indoor vital workers (% of labour force)"),
]
ESSENTIAL_STACK = [
    ("a", "%Essential Workers", "Total essential workers (% of labour force)"),
    ("b", "%Indoor Essential Workers", "Indoor essential workers (% of labour force)"),
]
GRID_2X2 = [
    ("a", "%Essential Workers", "Total essential workers (% of labour force)"),
    ("b", "%Indoor Essential Workers", "Indoor essential workers (% of labour force)"),
    ("c", "%Vital Workers", "Total vital workers (% of labour force)"),
    ("d", "%Indoor Vital Workers", "Indoor vital workers (% of labour force)"),
]


def _pct_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    plot_df = df[["Country Code", col]].copy()
    plot_df["pct"] = plot_df[col] * 100.0
    return plot_df.dropna(subset=["pct"])


def plot_stacked_pair(
    df: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """Two stacked maps sharing one colorbar (option 1)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.09, hspace=0.12)

    mappable = None
    for ax, (letter, col, title) in zip(axes, panels):
        mappable = draw_world_choropleth(
            ax,
            _pct_frame(df, col),
            iso_col="Country Code",
            value_col="pct",
            cmap=CMAP,
            vmin=VMIN,
            vmax=VMAX,
        )
        label_panel(ax, letter)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

    cax = fig.add_axes([0.24, 0.03, 0.52, 0.022])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(LEGEND_LABEL, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid global savefig.bbox="tight" so the manual colorbar placement sticks.
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=None)
    plt.close(fig)


def plot_2x2_grid(
    df: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """Four-panel map with one shared colorbar (option 2)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.9))
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.93, bottom=0.09, hspace=0.14, wspace=0.03
    )
    flat_axes = axes.ravel()

    mappable = None
    for ax, (letter, col, title) in zip(flat_axes, panels):
        mappable = draw_world_choropleth(
            ax,
            _pct_frame(df, col),
            iso_col="Country Code",
            value_col="pct",
            cmap=CMAP,
            vmin=VMIN,
            vmax=VMAX,
        )
        label_panel(ax, letter)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)

    cax = fig.add_axes([0.30, 0.03, 0.40, 0.022])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(LEGEND_LABEL, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=None)
    plt.close(fig)


def main(output_dir: Path) -> None:
    apply_allfed_style()
    ew_path = ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv"
    df = pd.read_csv(ew_path).dropna(subset=["Country Code"])

    vital_path = output_dir / "PctVitalWorkers_Manuscript.png"
    plot_stacked_pair(df, VITAL_STACK, vital_path)
    print(f"Wrote {vital_path}")

    essential_path = output_dir / "PctEssentialWorkers_Manuscript.png"
    plot_stacked_pair(df, ESSENTIAL_STACK, essential_path)
    print(f"Wrote {essential_path}")

    grid_path = output_dir / "PctWorkersByCountry_Manuscript_2x2.png"
    plot_2x2_grid(df, GRID_2X2, grid_path)
    print(f"Wrote {grid_path}")


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
