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
from matplotlib.patches import Rectangle
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    CR_BOXES_PRIORITIZED_RESULTS,
    PACS_PRIORITIZED_RESULTS,
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
LEGEND_FRAMEON = False
# Panel-letter position for ScenarioCoverage_Manuscript_two_panel (axes coordinates).
SCENARIO_COVERAGE_PANEL_LABEL_XY = (-0.08, 1.129)

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
    path = PACS_PRIORITIZED_RESULTS / f"coverage_{label}_scenario{scenario}.csv"
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
        PACS_PRIORITIZED_RESULTS / f"weekly_ecadr_by_country_scenario{scenario}.csv",
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
        PACS_PRIORITIZED_RESULTS / "requirements_by_region.csv", index_col="region"
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


def _draw_scenario_coverage_ax(
    ax,
    by_scenario,
    essential_level,
    last_week,
    title=None,
    ylim=None,
    show_legend=True,
    show_xlabel=True,
):
    """
    Draw global scenario coverage on one axes.

    Arguments:
        ax (matplotlib.axes.Axes): Axes to draw on.
        by_scenario (dict): Global vital coverage by scenario.
        essential_level (float): Essential requirement as % of vital.
        last_week (int): Last week on the x-axis.
        title (str or None): Axes title.
        ylim (tuple or None): y-axis limits, or None for autoscale.
        show_legend (bool): Whether to draw the legend.
        show_xlabel (bool): Whether to draw the x-axis label.
    """
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
        label=("Requirements range (indoor vital to indoor essential)"),
    )

    ax.set_xlim(1, last_week)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_xlabel:
        ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    ax.grid(True, linestyle="--", alpha=0.4)
    right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y * 100.0 / essential_level,
            lambda y: y * essential_level / 100.0,
        ),
    )
    right.set_ylabel("% of indoor essential worker requirement")
    if title:
        ax.set_title(title, fontweight="bold")
    if show_legend:
        ax.legend(
            fontsize=9,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=LEGEND_FRAMEON,
            framealpha=0.9,
        )


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
    title = (
        "Global filtration supply against workforce requirements\n"
        f"medians with {interval:.0f}% uncertainty intervals"
    )
    _draw_scenario_coverage_ax(
        ax, by_scenario, essential_level, last_week, title=title, ylim=(0, 30)
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_scenario_coverage_two_panel(output_path: Path) -> None:
    """
    Global coverage over time: full y-range above, 0–30% zoom below.

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
    zoom_ylim = (0, 30)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    title_line = f"Medians ± {interval:.0f}% uncertainty intervals"
    _draw_scenario_coverage_ax(
        ax_top,
        by_scenario,
        essential_level,
        last_week,
        title=(
            "Global filtration supply against workforce requirements (full scale)\n"
            f"{title_line}"
        ),
        show_legend=False,
        show_xlabel=False,
    )
    ax_top.add_patch(
        Rectangle(
            (1, zoom_ylim[0]),
            last_week - 1,
            zoom_ylim[1] - zoom_ylim[0],
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="0.25",
            zorder=10,
        )
    )
    _draw_scenario_coverage_ax(
        ax_bottom,
        by_scenario,
        essential_level,
        last_week,
        title=(
            "Global filtration supply against workforce requirements (0–30%)\n"
            f"{title_line}"
        ),
        show_legend=False,
        ylim=zoom_ylim,
    )
    label_x, label_y = SCENARIO_COVERAGE_PANEL_LABEL_XY
    label_panel(ax_top, "a", x=label_x, y=label_y)
    label_panel(ax_bottom, "b", x=label_x, y=label_y)
    fig.subplots_adjust(hspace=0.35, bottom=0.12)
    fig.legend(
        *ax_top.get_legend_handles_labels(),
        fontsize=9,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=LEGEND_FRAMEON,
        framealpha=0.9,
    )
    _save_figure(fig, output_path)


def _stackplot_channel_colors(swap_repurposed_cr_baghouse=False):
    """
    Default stackplot colours for the supply channels.

    Arguments:
        swap_repurposed_cr_baghouse (bool): If True, swap repurposed CR box and
            repurposed baghouse colours so CR box channels are both green.

    Returns:
        list: One colour per channel in CHANNEL_LABELS order.
    """
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    channel_colors = colors[: len(CHANNEL_LABELS)]
    if swap_repurposed_cr_baghouse:
        channel_colors[1], channel_colors[5] = channel_colors[5], channel_colors[1]
    return channel_colors


def plot_stacked_channels(
    output_path: Path,
    scenario: int,
    results_dir: Path = PACS_PRIORITIZED_RESULTS,
    title_suffix: str = "",
    swap_repurposed_cr_baghouse_colors: bool = False,
) -> None:
    """
    Global coverage over time, broken down by supply channel.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        results_dir (Path): Directory with ``ecadr_by_channel`` CSVs.
        title_suffix (str): Extra line for the figure title.
        swap_repurposed_cr_baghouse_colors (bool): Swap repurposed CR box and
            repurposed baghouse colours.
    """
    channels = pd.read_csv(
        results_dir / f"ecadr_by_channel_scenario{scenario}.csv", index_col="week"
    )
    requirement = pd.read_csv(
        results_dir / "requirements_by_region.csv", index_col="region"
    ).loc["Global", "indoor_vital_ecadr_l_per_s"]
    shares = 100.0 * channels[list(CHANNEL_LABELS)] / requirement

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        shares.index,
        *[shares[name] for name in CHANNEL_LABELS],
        labels=list(CHANNEL_LABELS.values()),
        colors=_stackplot_channel_colors(swap_repurposed_cr_baghouse_colors),
        alpha=0.75,
    )
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    title = f"Global filtration supply by source\n{SCENARIO_LABELS[scenario]}"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=LEGEND_FRAMEON,
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

    fig, (ax_supply, ax_coverage) = plt.subplots(
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
    label_panel(ax_supply, "a")
    ax_supply.set_title(
        # f"Filtration supply by UN region (week {week})",
        "Filtration supply by UN region after 3 months",
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
    label_panel(ax_coverage, "b")
    ax_coverage.set_title(
        # f"Indoor vital workers covered by filtration (week {week})",
        "Indoor vital workers covered by filtration after 3 months",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )

    # The panels are on different scales, so each carries its own bar
    add_horizontal_colorbar(fig, supply_mappable, ax_supply, "eCADR (billion L/s)")
    add_horizontal_colorbar(
        fig, coverage_mappable, ax_coverage, "% of indoor vital workers covered"
    )

    _save_figure(fig, output_path)


def main(output_dir: Path, scenario: int, week: int) -> None:
    apply_allfed_style()

    scenario_path = output_dir / "ScenarioCoverage_Manuscript.png"
    plot_scenario_coverage(scenario_path)
    print(f"Wrote {scenario_path}")

    scenario_two_panel_path = output_dir / "ScenarioCoverage_Manuscript_two_panel.png"
    plot_scenario_coverage_two_panel(scenario_two_panel_path)
    print(f"Wrote {scenario_two_panel_path}")

    stacked_path = output_dir / "Global_stacked_cadr.png"
    plot_stacked_channels(
        stacked_path, scenario, swap_repurposed_cr_baghouse_colors=True
    )
    print(f"Wrote {stacked_path}")

    cr_path = output_dir / "Global_stacked_cadr_CR_boxes_prioritized.png"
    plot_stacked_channels(
        cr_path,
        scenario,
        results_dir=CR_BOXES_PRIORITIZED_RESULTS,
        title_suffix="CR boxes prioritized (panel filters diverted from PACs)",
    )
    print(f"Wrote {cr_path}")

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
