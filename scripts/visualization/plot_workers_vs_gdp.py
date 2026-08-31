#!/usr/bin/env python3
"""Scatter plots of worker shares vs GDP per capita (PPP) with correlations.

Produces:
  - labour-force shares (essential / indoor essential / vital / indoor vital)
    vs log GDP per capita
  - Food as a share of essential / vital (and indoor) workforce vs log GDP

Also writes a correlation summary CSV under results/essential_workers/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    SAVE_DPI,
    VISUALIZATIONS_RESULTS,
    apply_allfed_style,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import essential_workers as ew  # noqa: E402

LF_PANELS = [
    ("a", "%Essential Workers", "Essential workers (% of labour force)"),
    ("b", "%Indoor Essential Workers", "Indoor essential (% of labour force)"),
    ("c", "%Vital Workers", "Vital workers (% of labour force)"),
    ("d", "%Indoor Vital Workers", "Indoor vital (% of labour force)"),
]
FOOD_PANELS = [
    ("a", "Food % of Essential Workers", "Food share of essential workforce"),
    ("b", "Food % of Indoor Essential Workers", "Food share of indoor essential"),
    ("c", "Food % of Vital Workers", "Food share of vital workforce"),
    ("d", "Food % of Indoor Vital Workers", "Food share of indoor vital"),
]


def _annotation(summary: pd.DataFrame, column: str) -> str:
    row = summary.loc[summary["column"] == column].iloc[0]
    return (
        f"Spearman ρ = {row['Spearman ρ']:.2f}\n"
        f"p = {row['Spearman p']:.2e}\n"
        f"n = {int(row['n'])}"
    )


def _scatter_grid(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    output_path: Path,
    *,
    gdp_col: str,
    y_as_percent: bool,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.32, wspace=0.25
    )
    x = merged[gdp_col]

    for ax, (letter, col, title) in zip(axes.ravel(), panels):
        y = merged[col] * (100.0 if y_as_percent else 1.0)
        mask = x.notna() & y.notna() & (x > 0)
        ax.scatter(
            x[mask],
            y[mask],
            s=18,
            alpha=0.75,
            edgecolors="none",
            zorder=2,
        )
        # Ordinary least-squares trend on log-x for visual guidance only.
        log_x = np.log(x[mask].to_numpy(dtype=float))
        y_vals = y[mask].to_numpy(dtype=float)
        if log_x.size >= 3:
            slope, intercept = np.polyfit(log_x, y_vals, 1)
            x_line = np.geomspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(
                x_line,
                slope * np.log(x_line) + intercept,
                color="C1",
                lw=1.8,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(
            0.02,
            0.98,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.text(
            0.98,
            0.98,
            _annotation(summary, col),
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, lw=0),
        )
        ax.set_xlabel("GDP per capita, PPP (int'l $)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI)
    plt.close(fig)


def main(output_dir: Path) -> None:
    apply_allfed_style()
    lf_df = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv")
    group_df = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByGroup.csv")
    merged, summary = ew.summarize_worker_shares_vs_gdp(lf_df, group_df)

    corr_path = ESSENTIAL_WORKERS_RESULTS / "WorkerShares_vs_GDP_Correlations.csv"
    summary.to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}")
    print(summary.to_string(index=False))

    merged_path = ESSENTIAL_WORKERS_RESULTS / "WorkerShares_vs_GDP.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")

    lf_path = output_dir / "WorkerShares_vs_GDP_PPP.png"
    _scatter_grid(
        merged,
        summary,
        LF_PANELS,
        lf_path,
        gdp_col=ew.GDP_PPP_COL,
        y_as_percent=True,
        ylabel="% of labour force",
    )
    print(f"Wrote {lf_path}")

    food_path = output_dir / "FoodShareOfWorkforce_vs_GDP_PPP.png"
    _scatter_grid(
        merged,
        summary,
        FOOD_PANELS,
        food_path,
        gdp_col=ew.GDP_PPP_COL,
        y_as_percent=True,
        ylabel="% of essential / vital workforce",
    )
    print(f"Wrote {food_path}")


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
