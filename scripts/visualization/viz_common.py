"""Shared helpers for ALLFED-styled matplotlib visualizations."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Re-exported so the figure scripts have one place to import paths from
from paths import (  # noqa: E402,F401
    ESSENTIAL_WORKERS_RESULTS,
    SCALE_UP_RESULTS,
    VISUALIZATIONS_RESULTS,
)


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
    """Natural Earth country polygons (110m), in Winkel Tripel."""
    world = gpd.read_file(NATURAL_EARTH_110M_URL)
    iso = world["ISO_A3"].where(
        world["ISO_A3"].notna() & (world["ISO_A3"] != "-99"),
        world["ISO_A3_EH"],
    )
    world = world.assign(iso_a3=iso)
    world = world[world["iso_a3"].notna() & (world["iso_a3"] != "-99")].copy()
    return world.to_crs(WINKEL_TRIPEL)


@lru_cache(maxsize=1)
def load_map_border() -> gpd.GeoDataFrame:
    """ALLFED map outline. Already in Winkel Tripel despite its stated CRS."""
    return gpd.read_file(ALLFED_MAP_BORDER_URL)


def label_panel(ax: plt.Axes, letter: str) -> None:
    """Mark a panel with its letter, inside the axes so titles stay clear."""
    ax.text(
        0.01,
        0.99,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_world_choropleth(
    ax: plt.Axes,
    data: pd.DataFrame,
    iso_col: str,
    value_col: str,
    *,
    cmap: str = "viridis",
    alpha: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Draw a choropleth onto ``ax`` without a colorbar; return a ScalarMappable.

    The view is trimmed to inhabited latitudes, since Antarctica never carries
    a value and the empty polar bands waste space in multi-panel figures.
    """
    merged = load_world().merge(
        data[[iso_col, value_col]].dropna(subset=[value_col]),
        left_on="iso_a3",
        right_on=iso_col,
        how="left",
    )
    merged.plot(
        column=value_col,
        ax=ax,
        legend=False,
        missing_kwds={"color": "lightgrey"},
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="dimgray",
        linewidth=0.1,
        alpha=alpha,
    )
    load_map_border().plot(ax=ax, edgecolor="black", linewidth=0.1, facecolor="none")
    ax.set_axis_off()
    # _, ymin, _, ymax = merged[merged.iso_a3 != "ATA"].total_bounds
    # ax.set_ylim(ymin, ymax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)
