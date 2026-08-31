#!/usr/bin/env python3
"""Stacked composition of essential / vital workforces by occupational group.

Produces a 2×2 figure (essential, indoor essential, vital, indoor vital) with
global worker-weighted shares, and writes composition CSVs if missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

PANELS = [
    ("a", "% of Essential Workers", "Essential workforce"),
    ("b", "% of Indoor Essential Workers", "Indoor essential workforce"),
    ("c", "% of Vital Workers", "Vital workforce"),
    ("d", "% of Indoor Vital Workers", "Indoor vital workforce"),
]


def _ensure_composition_csvs(group_df: pd.DataFrame) -> pd.DataFrame:
    global_df = ew.summarize_group_composition(group_df)
    global_df.to_csv(
        ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByGroupComposition_Global.csv",
        index=False,
    )
    ew.summarize_group_composition(group_df, by="Region").to_csv(
        ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByGroupComposition_ByRegion.csv",
        index=False,
    )
    ew.summarize_group_composition(group_df, by="Country").to_csv(
        ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByGroupComposition_ByCountry.csv",
        index=False,
    )
    return global_df


def plot_global_composition(global_df: pd.DataFrame, output_path: Path) -> None:
    groups = list(global_df["occupational_group"])
    # Distinct qualitative colors (avoid default cycle collisions).
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(groups))]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    fig.subplots_adjust(
        left=0.07, right=0.78, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25
    )

    for ax, (letter, col, title) in zip(axes.ravel(), PANELS):
        shares = global_df[col].to_numpy(dtype=float) * 100.0
        left = 0.0
        for share, color in zip(shares, colors):
            ax.barh(0, share, left=left, height=0.55, color=color, edgecolor="none")
            left += share
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of workforce", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(
            0.01,
            1.12,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )
        # Percent labels for large segments.
        left = 0.0
        for share in shares:
            if share >= 8.0:
                ax.text(
                    left + share / 2.0,
                    0.0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            left += share

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in zip(groups, colors)
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False,
        fontsize=9,
        title="Occupational group",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI)
    plt.close(fig)


def main(output_dir: Path) -> None:
    apply_allfed_style()
    group_path = ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByGroup.csv"
    group_df = pd.read_csv(group_path)
    global_df = _ensure_composition_csvs(group_df)

    print("Global composition (% of category workforce):")
    display = global_df[["occupational_group", *ew.GROUP_COMPOSITION_SHARE_COLS]].copy()
    for col in ew.GROUP_COMPOSITION_SHARE_COLS:
        display[col] = (display[col] * 100.0).round(1)
    print(display.to_string(index=False))

    out = output_dir / "GroupComposition_Global.png"
    plot_global_composition(global_df, out)
    print(f"Wrote {out}")


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
