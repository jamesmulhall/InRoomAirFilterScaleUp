#!/usr/bin/env python3
"""Manuscript figures for the filtration scale-up (ALLFED style).

Reads the scenario results written by ``src/scale_up_model.py`` and produces:
  - global coverage over time under all three scenarios, with uncertainty
  - the same coverage broken down by supply channel
  - a two-panel map of coverage by UN region at a chosen week
  - a two-panel map of regional eCADR supply and vital coverage at a chosen week
  - separate maps for indoor vital and indoor essential coverage

Coverage is expressed as a share of the indoor vital worker requirement. The
indoor essential requirement is larger, so full essential coverage sits above
100 percent on that scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    SCALE_UP_RESULTS,
    SAVE_DPI,
    VISUALIZATIONS_RESULTS,
    add_horizontal_colorbar,
    apply_allfed_style,
    draw_world_choropleth,
    expand_regions_to_countries,
    label_panel,
)

# cmasher colormap names; change these to try other palettes from the library.
# CMAP_RANGE drops the black tip of arctic_r so the highest values stay navy.
REGION_COVERAGE_CMAP = "arctic_r"
REGION_SUPPLY_CMAP = "arctic_r"
CMAP_RANGE = (0.0, 0.95)
REGION_COVERAGE_VMAX = 100.0
REGION_COVERAGE_ALPHA = 0.8
# Extra vertical gap between stacked map panels, as a fraction of panel height.
# Constrained layout's default is 0.02; raise this to pull the panels apart.
PANEL_HSPACE = 0.10

# Supply channels, in stacking order, with manuscript labels
CHANNEL_LABELS = {
    "cr_box": "Corsi-Rosenthal boxes, new",
    "repurposed_cr_box": "Corsi-Rosenthal boxes, repurposed filters",
    "pac": "Portable air cleaners, new",
    "repurposed_pac": "Portable air cleaners, repurposed",
    "baghouse": "Coal baghouse bags, new",
    "repurposed_baghouse": "Coal baghouse bags, repurposed",
}

SCENARIO_LABELS = {
    3: "Scenario 1: growth based on N95s during COVID-19",
    2: "Scenario 2: growth capped by meltblown supply",
    1: "Scenario 3: higher factory utilisation only",
}


def load_coverage(label: str, scenario: int) -> pd.DataFrame:
    """
    Read one coverage table.

    Arguments:
        label (str): "vital" or "essential".
        scenario (int): 1, 2 or 3.

    Returns:
        pandas.DataFrame: Coverage by region and week, as percentages.
    """
    path = SCALE_UP_RESULTS / f"coverage_{label}_scenario{scenario}.csv"
    df = pd.read_csv(path)
    for column in ["coverage_median", "coverage_lower", "coverage_upper"]:
        df[column] *= 100.0
    return df


def load_regional_ecadr(scenario: int, week: int) -> pd.DataFrame:
    """
    Sum median weekly eCADR to each UN region.

    Arguments:
        scenario (int): 1, 2 or 3.
        week (int): Week to sum.

    Returns:
        pandas.DataFrame: Columns region and ecadr_l_per_s.
    """
    weekly = pd.read_csv(
        SCALE_UP_RESULTS / f"weekly_ecadr_by_country_scenario{scenario}.csv",
        index_col=0,
    ).reset_index(names="Country Name")
    regions = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv")[
        ["Country Name", "Region"]
    ]
    merged = weekly.merge(regions, on="Country Name")
    column = str(week)
    if column not in merged.columns:
        raise ValueError(f"Week {week} not in weekly eCADR output")
    return (
        merged.groupby("Region", as_index=False)[column]
        .sum()
        .rename(columns={"Region": "region", column: "ecadr_l_per_s"})
    )


def load_requirements() -> pd.DataFrame:
    """
    Read the eCADR requirement of each region and of the world.

    Returns:
        pandas.DataFrame: Indexed by region, in L/s.
    """
    return pd.read_csv(
        SCALE_UP_RESULTS / "requirements_by_region.csv", index_col="region"
    )


def essential_requirement_level(requirements: pd.DataFrame) -> float:
    """
    Global indoor essential requirement, as a percentage of the vital one.

    Arguments:
        requirements (pandas.DataFrame): Output of load_requirements.

    Returns:
        float: Percentage, above 100 because the essential workforce is wider.
    """
    return 100.0 * (
        requirements.loc["Global", "indoor_essential_ecadr_l_per_s"]
        / requirements.loc["Global", "indoor_vital_ecadr_l_per_s"]
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_coverage(output_path: Path) -> None:
    """
    Global coverage over time under all three scenarios.

    Each scenario gets its median and uncertainty interval. The band at the top
    marks the range between fully covering indoor vital workers and fully
    covering the wider indoor essential workforce.

    Arguments:
        output_path (Path): PNG to write.
    """
    requirements = load_requirements()
    essential_level = essential_requirement_level(requirements)
    by_scenario = {}
    for scenario in SCENARIO_LABELS:
        df = load_coverage("vital", scenario)
        by_scenario[scenario] = df[df.region == "Global"]
    interval = by_scenario[1].interval_percent.iloc[0]
    last_week = by_scenario[1].week.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    # Scenario 3 is dashed so it stays visible where the meltblown cap does not
    # bind and scenario 2 lies on top of it
    for (scenario, label), style in zip(SCENARIO_LABELS.items(), ["-", "-", "-"]):
        df = by_scenario[scenario]
        (line,) = ax.plot(
            df.week, df.coverage_median, linewidth=2, linestyle=style, label=label
        )
        ax.fill_between(
            df.week,
            df.coverage_lower,
            df.coverage_upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )

    ax.axhspan(
        100.0,
        essential_level,
        color="dimgray",
        alpha=0.15,
        label=(
            "Full coverage: indoor vital (100%) to "
            f"indoor essential workers ({essential_level:.0f}%)"
        ),
    )

    ax.set_xlim(1, last_week)
    # ax.set_ylim(0, essential_level * 1.05)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    ax.set_title(
        "Global filtration supply against workforce requirements\n"
        f"medians with {interval:.0f}% uncertainty intervals",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y * 100.0 / essential_level,
            lambda y: y * essential_level / 100.0,
        ),
    )
    right.set_ylabel("% of indoor essential worker requirement")
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


def plot_stacked_channels(output_path: Path, scenario: int) -> None:
    """
    Global coverage over time, broken down by supply channel.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
    """
    channels = pd.read_csv(
        SCALE_UP_RESULTS / f"ecadr_by_channel_scenario{scenario}.csv", index_col="week"
    )
    requirement = load_requirements().loc["Global", "indoor_vital_ecadr_l_per_s"]
    shares = 100.0 * channels[list(CHANNEL_LABELS)] / requirement

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        shares.index,
        *[shares[name] for name in CHANNEL_LABELS],
        labels=list(CHANNEL_LABELS.values()),
        alpha=0.75,
    )
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    ax.set_title(
        f"Global filtration supply by source\n{SCENARIO_LABELS[scenario]}",
        fontweight="bold",
    )
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


def plot_single_region_coverage_map(
    output_path: Path,
    scenario: int,
    week: int,
    label: str,
    title: str,
) -> None:
    """
    One world map of coverage by UN region at a chosen week.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        week (int): Week to map.
        label (str): "vital" or "essential".
        title (str): Figure title.
    """
    df = load_coverage(label, scenario)
    at_week = df[(df.week == week) & (df.region != "Global")]
    if at_week.empty:
        raise ValueError(f"No {label} coverage at week {week}")

    data = expand_regions_to_countries(at_week, "region", "coverage_median")
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    mappable = draw_world_choropleth(
        ax,
        data,
        iso_col="Country Code",
        value_col="coverage_median",
        cmap=REGION_COVERAGE_CMAP,
        cmap_range=CMAP_RANGE,
        vmin=0.0,
        vmax=REGION_COVERAGE_VMAX,
        alpha=REGION_COVERAGE_ALPHA,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    add_horizontal_colorbar(fig, mappable, ax, "% of indoor workers covered")
    _save_figure(fig, output_path)


def plot_region_coverage_maps(output_path: Path, scenario: int, week: int) -> None:
    """
    Coverage by UN region at one week: indoor essential above, indoor vital below.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        week (int): Week to map.
    """
    panels = []
    for letter, label, name in [
        ("a", "essential", "Indoor essential workers"),
        ("b", "vital", "Indoor vital workers"),
    ]:
        df = load_coverage(label, scenario)
        at_week = df[(df.week == week) & (df.region != "Global")]
        if at_week.empty:
            raise ValueError(f"No {label} coverage at week {week}")
        panels.append(
            (
                letter,
                expand_regions_to_countries(at_week, "region", "coverage_median"),
                f"{name} covered by filtration (week {week})",
            )
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), layout="constrained")

    mappable = None
    for ax, (letter, data, title) in zip(axes, panels):
        mappable = draw_world_choropleth(
            ax,
            data,
            iso_col="Country Code",
            value_col="coverage_median",
            cmap=REGION_COVERAGE_CMAP,
            cmap_range=CMAP_RANGE,
            vmin=0.0,
            vmax=REGION_COVERAGE_VMAX,
            alpha=REGION_COVERAGE_ALPHA,
        )
        label_panel(ax, letter)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    # Both panels share one scale, so one bar spans them
    add_horizontal_colorbar(fig, mappable, list(axes), "% of indoor workers covered")
    _save_figure(fig, output_path)


def plot_supply_and_coverage_maps(output_path: Path, scenario: int, week: int) -> None:
    """
    Two-panel map: regional eCADR supply above, indoor vital coverage below.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        week (int): Week to map.
    """
    supply = load_regional_ecadr(scenario, week)
    supply = supply[supply.region != "Global"].copy()
    supply["ecadr_billion_l_per_s"] = supply.ecadr_l_per_s / 1e9
    supply_map = expand_regions_to_countries(supply, "region", "ecadr_billion_l_per_s")

    coverage = load_coverage("vital", scenario)
    at_week = coverage[(coverage.week == week) & (coverage.region != "Global")]
    if at_week.empty:
        raise ValueError(f"No vital coverage at week {week}")
    coverage_map = expand_regions_to_countries(at_week, "region", "coverage_median")

    fig, (ax_coverage, ax_supply) = plt.subplots(
        2, 1, figsize=(10, 9), layout="constrained"
    )
    fig.set_constrained_layout_pads(hspace=PANEL_HSPACE)

    supply_mappable = draw_world_choropleth(
        ax_supply,
        supply_map,
        iso_col="Country Code",
        value_col="ecadr_billion_l_per_s",
        cmap=REGION_SUPPLY_CMAP,
        cmap_range=CMAP_RANGE,
        vmin=0.0,
        alpha=REGION_COVERAGE_ALPHA,
    )
    label_panel(ax_supply, "b")
    ax_supply.set_title(
        f"Filtration supply by UN region (week {week})",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )

    coverage_mappable = draw_world_choropleth(
        ax_coverage,
        coverage_map,
        iso_col="Country Code",
        value_col="coverage_median",
        cmap=REGION_COVERAGE_CMAP,
        cmap_range=CMAP_RANGE,
        vmin=0.0,
        vmax=REGION_COVERAGE_VMAX,
        alpha=REGION_COVERAGE_ALPHA,
    )
    label_panel(ax_coverage, "a")
    ax_coverage.set_title(
        f"Indoor vital workers covered by filtration (week {week})",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )

    # The panels are on different scales, so each carries its own bar
    add_horizontal_colorbar(
        fig, supply_mappable, ax_supply, "eCADR (billion L/s)"
    )
    add_horizontal_colorbar(
        fig, coverage_mappable, ax_coverage, "% of indoor vital workers covered"
    )

    _save_figure(fig, output_path)


def main(output_dir: Path, scenario: int, week: int) -> None:
    apply_allfed_style()

    scenario_path = output_dir / "ScenarioCoverage_Manuscript.png"
    plot_scenario_coverage(scenario_path)
    print(f"Wrote {scenario_path}")

    stacked_path = output_dir / "Global_stacked_cadr.png"
    plot_stacked_channels(stacked_path, scenario)
    print(f"Wrote {stacked_path}")

    map_path = output_dir / f"FiltrationCoverageByRegion_Manuscript_Week{week}.png"
    plot_region_coverage_maps(map_path, scenario, week)
    print(f"Wrote {map_path}")

    supply_path = output_dir / f"FiltrationSupplyAndCoverage_Manuscript_Week{week}.png"
    plot_supply_and_coverage_maps(supply_path, scenario, week)
    print(f"Wrote {supply_path}")

    vital_path = output_dir / f"FiltrationCoverageVital_Manuscript_Week{week}.png"
    plot_single_region_coverage_map(
        vital_path,
        scenario,
        week,
        "vital",
        f"Indoor vital workers covered by filtration (week {week})",
    )
    print(f"Wrote {vital_path}")

    essential_path = (
        output_dir / f"FiltrationCoverageEssential_Manuscript_Week{week}.png"
    )
    plot_single_region_coverage_map(
        essential_path,
        scenario,
        week,
        "essential",
        f"Indoor essential workers covered by filtration (week {week})",
    )
    print(f"Wrote {essential_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATIONS_RESULTS,
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Scenario for the stacked figure and the maps",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=13,
        help="Week to map (13 is three months)",
    )
    args = parser.parse_args()
    main(args.output_dir, args.scenario, args.week)
