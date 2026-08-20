"""Shared helpers for ALLFED-styled matplotlib visualizations."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALLFED_STYLE_URL = (
    "https://raw.githubusercontent.com/allfed/"
    "ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)
SAVE_DPI = 300
NATURAL_EARTH_110M_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/" "ne_110m_admin_0_countries.zip"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from paths import (  # noqa: E402
    ESSENTIAL_WORKERS_RESULTS,
    SCALE_UP_RESULTS,
    VISUALIZATIONS_RESULTS,
)

_UNCERTAINTY_RE = re.compile(
    r"^\(?(?P<nom>[\d.]+)\+/-(?P<std>[\d.]+)\)?(?P<exp>[eE][+-]?\d+)?$"
)


def apply_allfed_style() -> None:
    """Apply ALLFED matplotlib style and default figure settings."""
    plt.style.use(ALLFED_STYLE_URL)
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.dpi"] = SAVE_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI
    plt.rcParams["savefig.bbox"] = "tight"


def nominal_value(cell) -> float:
    """Extract nominal float from ufloat, number, or uncertainty string."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    if hasattr(cell, "nominal_value"):
        return float(cell.nominal_value)
    if isinstance(cell, (int, float, np.floating)):
        return float(cell)
    text = str(cell).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    match = _UNCERTAINTY_RE.match(text)
    if match:
        nom = float(match.group("nom"))
        exp = match.group("exp")
        if exp:
            nom *= float(f"1{exp}")
        return nom
    return np.nan


def week_column(week: int) -> Union[int, str]:
    """Column label for week *n* in scale-up tables (after two prefix columns)."""
    return week + 2


def load_scale_up_table(path: Path) -> pd.DataFrame:
    """Load scale-up table from pickle (preferred) or CSV."""
    pkl_path = path.with_suffix(".pkl")
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    df = pd.read_csv(path, index_col=0)
    df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
    return df


def load_country_iso_map(
    ew_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Country name → ISO3 from essential workers output."""
    path = ew_csv or (ESSENTIAL_WORKERS_RESULTS / "EssentialWorkersByCountry.csv")
    df = pd.read_csv(path)[["Country Name", "Country Code"]].dropna()
    return df.drop_duplicates(subset="Country Name")


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


def load_world() -> gpd.GeoDataFrame:
    """Natural Earth country polygons (110m)."""
    world = gpd.read_file(NATURAL_EARTH_110M_URL)
    iso = world["ISO_A3"].where(
        world["ISO_A3"].notna() & (world["ISO_A3"] != "-99"),
        world["ISO_A3_EH"],
    )
    world = world.assign(iso_a3=iso)
    world = world[world["iso_a3"].notna() & (world["iso_a3"] != "-99")].copy()
    return world


def plot_world_choropleth(
    data: pd.DataFrame,
    iso_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    *,
    cmap: str = "viridis",
    alpha: float = 1.0,
    legend_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hide_internal_borders: bool = False,
) -> None:
    """Merge ISO3 values onto world map and save PNG."""
    world = load_world()
    merged = world.merge(
        data[[iso_col, value_col]].dropna(subset=[value_col]),
        left_on="iso_a3",
        right_on=iso_col,
        how="left",
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_kwargs: dict = {
        "column": value_col,
        "ax": ax,
        "legend": True,
        "missing_kwds": {"color": "lightgrey", "label": "No data"},
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "edgecolor": "dimgray",
        "linewidth": 0.1,
        "legend_kwds": {
            "label": legend_label or value_col,
            "orientation": "horizontal",
            "shrink": 0.6,
        },
    }
    if hide_internal_borders:
        plot_kwargs["edgecolor"] = "face"
        plot_kwargs["linewidth"] = 0.1
    merged.plot(**plot_kwargs)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def get_series(
    df: pd.DataFrame, region: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (weeks, values, errors) for a region row (notebook-compatible)."""
    row = df.loc[region]
    series = row.iloc[2:]
    time = np.arange(len(series))
    values = np.array([nominal_value(v) for v in series])
    errors = np.array([series_std_dev(v) for v in series])
    return time, values, errors


def series_std_dev(cell) -> float:
    """Standard deviation from ufloat or uncertainty string; else 0."""
    if hasattr(cell, "std_dev"):
        return float(cell.std_dev)
    text = str(cell).strip()
    match = _UNCERTAINTY_RE.match(text)
    if match:
        std = float(match.group("std"))
        exp = match.group("exp")
        if exp:
            std *= float(f"1{exp}")
        return std
    return 0.0


def add_worker_range_band(
    ax: plt.Axes,
    df: pd.DataFrame,
    region: str,
    *,
    cadrpp: float = 100.0,
    alpha: float = 0.15,
    color: str | None = None,
) -> tuple[float, float]:
    """Shade CADR band from indoor vital to indoor essential worker demand."""
    indoor_vital, indoor_essential = indoor_worker_bounds(df, region)
    cadr_lower = cadrpp * indoor_vital
    cadr_upper = cadrpp * indoor_essential
    span_kwargs = {"alpha": alpha}
    if color is not None:
        span_kwargs["color"] = color
    ax.axhspan(
        cadr_lower,
        cadr_upper,
        **span_kwargs,
        label=(
            f"Vital–essential worker range "
            f"({cadr_lower:.1e} – {cadr_upper:.1e} L/s)"
        ),
    )
    return cadr_lower, cadr_upper


def indoor_worker_bounds(df: pd.DataFrame, region: str) -> tuple[float, float]:
    """Indoor vital (lower) and indoor essential (upper) counts."""
    row = df.loc[region]
    return nominal_value(row.iloc[0]), nominal_value(row.iloc[1])
