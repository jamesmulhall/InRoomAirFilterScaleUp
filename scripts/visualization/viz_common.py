"""Shared helpers for ALLFED-styled matplotlib visualizations."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import cmasher as cm
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Colormap

ALLFED_STYLE_URL = (
    "https://raw.githubusercontent.com/allfed/"
    "ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)
SAVE_DPI = 300
NATURAL_EARTH_110M_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/" "ne_110m_admin_0_countries.zip"
)
ALLFED_MAP_BORDER_URL = (
    "https://raw.githubusercontent.com/ALLFED/ALLFED-map-border/main/border.geojson"
)
# Published ALLFED maps use Winkel Tripel, a compromise projection chosen for
# readability. It preserves neither area, angle nor distance, so it is for
# display only.
WINKEL_TRIPEL = "+proj=wintri"

# Set to True to draw the ALLFED map outline on every choropleth.
SHOW_MAP_BORDER = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Re-exported so the figure scripts have one place to import paths from
from paths import (  # noqa: E402,F401
    ESSENTIAL_WORKERS_RESULTS,
    SCALE_UP_RESULTS,
    VISUALIZATIONS_RESULTS,
)


def get_cmap(name: str, start: float = 0.0, stop: float = 1.0) -> Colormap:
    """
    Resolve a colormap by name, optionally keeping only part of it.

    Checks cmasher first, then matplotlib's registry. ``start`` and ``stop``
    are fractions of the original map, used to drop an end that is too dark
    or too light (cmasher's ``get_sub_cmap``).

    Arguments:
        name (str): Colormap name.
        start (float): Lower fraction of the original map to keep (0 to 1).
        stop (float): Upper fraction of the original map to keep (0 to 1).

    Returns:
        matplotlib.colors.Colormap: The resolved colormap.
    """
    cmap = getattr(cm, name) if hasattr(cm, name) else plt.get_cmap(name)
    if start == 0.0 and stop == 1.0:
        return cmap
    return cm.get_sub_cmap(cmap, start, stop)


def apply_allfed_style() -> None:
    """Apply ALLFED matplotlib style and default figure settings."""
    plt.style.use(ALLFED_STYLE_URL)
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.dpi"] = SAVE_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI
    plt.rcParams["savefig.bbox"] = "tight"


def load_country_region_map(
    ew_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Country ISO3 → UN region from essential workers output."""
    path = ew_csv or (ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv")
    return (
        pd.read_csv(path)[["Country Code", "Region"]]
        .dropna()
        .drop_duplicates(subset="Country Code")
    )


def expand_regions_to_countries(
    region_data: pd.DataFrame,
    region_col: str,
    value_col: str,
    *,
    ew_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Assign each country's ISO3 the value of its UN region."""
    country_regions = load_country_region_map(ew_csv)
    return country_regions.merge(
        region_data[[region_col, value_col]].dropna(subset=[value_col]),
        left_on="Region",
        right_on=region_col,
        how="inner",
    )[["Country Code", value_col]]


@lru_cache(maxsize=1)
def load_world() -> gpd.GeoDataFrame:
    """
    Natural Earth country polygons (110m), in Winkel Tripel.

    Antarctica is dropped: it never carries a value and its polygon takes up a
    quarter of the height of a world map.
    """
    world = gpd.read_file(NATURAL_EARTH_110M_URL)
    iso = world["ISO_A3"].where(
        world["ISO_A3"].notna() & (world["ISO_A3"] != "-99"),
        world["ISO_A3_EH"],
    )
    world = world.assign(iso_a3=iso)
    world = world[world["iso_a3"].notna() & (world["iso_a3"] != "-99")]
    world = world[world["iso_a3"] != "ATA"].copy()
    return world.to_crs(WINKEL_TRIPEL)


@lru_cache(maxsize=1)
def load_map_border() -> gpd.GeoDataFrame:
    """ALLFED map outline. Already in Winkel Tripel despite its stated CRS."""
    return gpd.read_file(ALLFED_MAP_BORDER_URL)


def label_panel(ax: plt.Axes, letter: str) -> None:
    """Mark a panel with its letter, inside the axes so titles stay clear."""
    ax.text(
        0.05,
        1.065,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_horizontal_colorbar(
    fig: plt.Figure,
    mappable,
    axes,
    label: str,
    *,
    shrink: float = 0.5,
):
    """
    Add a labelled horizontal colorbar under one or more map axes.

    The bar is attached to the axes rather than placed at fixed figure
    coordinates, so a figure using constrained layout keeps it clear of the
    titles below. Maps hold an equal aspect ratio, which leaves slack inside
    their axes, and fixed placement lands in that slack.

    Arguments:
        fig (matplotlib.figure.Figure): Figure holding the axes.
        mappable: What the colours came from.
        axes: One axes, or a list of axes to share the bar.
        label (str): Colorbar label.
        shrink (float): Bar length as a fraction of the axes width.

    Returns:
        matplotlib.colorbar.Colorbar: The colorbar drawn.
    """
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        location="bottom",
        shrink=shrink,
        aspect=45,
        pad=0.02,
    )
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    return cbar


def draw_world_choropleth(
    ax: plt.Axes,
    data: pd.DataFrame,
    iso_col: str,
    value_col: str,
    *,
    cmap: Union[str, Colormap] = "viridis",
    cmap_range: tuple = (0.0, 1.0),
    alpha: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Draw a choropleth onto ``ax`` without a colorbar; return a ScalarMappable.

    Any limit left unset is taken from the data, so the returned mappable and
    the map itself always share one scale. ``cmap_range`` is the fraction of
    the named colormap to keep, passed through to ``get_cmap``.
    """
    if isinstance(cmap, str):
        cmap_obj = get_cmap(cmap, cmap_range[0], cmap_range[1])
    else:
        cmap_obj = cmap
    merged = load_world().merge(
        data[[iso_col, value_col]].dropna(subset=[value_col]),
        left_on="iso_a3",
        right_on=iso_col,
        how="left",
    )
    if vmin is None:
        vmin = merged[value_col].min()
    if vmax is None:
        vmax = merged[value_col].max()
    merged.plot(
        column=value_col,
        ax=ax,
        legend=False,
        missing_kwds={"color": "lightgrey"},
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        edgecolor="dimgray",
        linewidth=0.1,
        alpha=alpha,
    )
    if SHOW_MAP_BORDER:
        load_map_border().plot(
            ax=ax, edgecolor="black", linewidth=0.1, facecolor="none"
        )
    ax.set_axis_off()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
