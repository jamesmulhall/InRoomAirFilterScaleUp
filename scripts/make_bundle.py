# make_bundle.py — export a single data_bundle.json for the static HTML app
import json
import pandas as pd
from pathlib import Path

# -----------------------------
# Fixed relative paths (exactly as requested)
# -----------------------------
file_path_main      = "../results/scale_up/Scale_up_output_MS.pkl"
file_path_percent   = "../results/scale_up/Scale_up_PERCENT_INDOOR_VITAL_MS.pkl"
file_path_cr_man    = "../results/scale_up/Scale_up_CR_MAN_MS.pkl"
file_path_cr_repur  = "../results/scale_up/Scale_up_CR_REPUR_MS.pkl"
file_path_coalbag   = "../results/scale_up/Scale_up_COALBAG_MS.pkl"
file_path_cr_stock  = "../results/scale_up/Scale_up_CR_STOCK.pkl"
file_path_portable = "../results/scale_up/Scale_up_PORTABLE_MS.pkl"
file_path_essential_workers_country = "../results/essential_workers/EssentialWorkersByCountry.csv"  # (not used in bundle, but kept for parity)

# --- add these 7 lines right after your file_path_* block ---
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
def _rp(rel: str) -> str:
    return str((_SCRIPT_DIR / rel).resolve())

file_path_main       = _rp(file_path_main)
file_path_percent    = _rp(file_path_percent)
file_path_cr_man     = _rp(file_path_cr_man)
file_path_cr_repur   = _rp(file_path_cr_repur)
file_path_coalbag    = _rp(file_path_coalbag)
file_path_cr_stock   = _rp(file_path_cr_stock)
file_path_portable = _rp(file_path_portable)
file_path_essential_workers_country= _rp(file_path_essential_workers_country)


# -----------------------------
# Constants (same UN regions as your notebook)
# -----------------------------
UNRegion_list = [
    'Australia and New Zealand','Caribbean','Central America','Central Asia',
    'Eastern Africa','Eastern Asia','Eastern Europe','Melanesia','Micronesia',
    'Middle Africa','Northern Africa','Northern America','Northern Europe',
    'Polynesia','South America','South-eastern Asia','Southern Africa',
    'Southern Asia','Southern Europe','Western Africa','Western Asia','Western Europe'
]

# -----------------------------
# Paths (repo root = one level above this script)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTFILE = REPO / "data_bundle.json"

def _abspath(rel: str) -> Path:
    return (SCRIPT_DIR / rel).resolve()

# -----------------------------
# Helpers
# -----------------------------
def coerce_numeric(v):
    """float(value) with support for uncertainties.ufloat; non-numeric -> None"""
    try:
        nv = getattr(v, "nominal_value", None)
        return float(nv if nv is not None else v)
    except Exception:
        return None

def load_pickle_table(rel_path: str) -> pd.DataFrame:
    """Read a pickle by relative path (from scripts/) and return DataFrame."""
    p = _abspath(rel_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_pickle(p)

def pick_week_columns(df: pd.DataFrame):
    """
    Robust week column selection:
    1) Prefer columns that can be cast to int (e.g., 1,2,3,... or '1','2',...).
    2) Fallback to df.columns[2:] if no numeric-castable columns found.
    Returns a list of column labels (as-is).
    """
    numeric_like = []
    for c in df.columns:
        try:
            int(c)
            numeric_like.append(c)
        except Exception:
            continue
    if numeric_like:
        return numeric_like
    # Fallback: first two columns are indoor-vital bounds, weeks from col index 2 onward
    return list(df.columns[2:])

def nominal_df_from_pickle(rel_path: str) -> tuple[pd.DataFrame, list]:
    """
    Notebook-equivalent: filter to UNRegion_list rows, keep only week columns,
    coerce to floats. Returns (df_nominal, week_cols_list).
    """
    df = load_pickle_table(rel_path)
    present = [r for r in UNRegion_list if r in df.index]
    if not present:
        raise KeyError(f"None of UNRegion_list found in {rel_path}. "
                       f"Index sample: {list(df.index[:10])}")
    df = df.loc[present]
    week_cols = pick_week_columns(df)
    df_nominal = df[week_cols].copy().applymap(coerce_numeric)
    return df_nominal, week_cols

def df_to_weekmap(df: pd.DataFrame) -> dict:
    """
    Convert region x week dataframe into { region: { '1': value, '2': value, ... } }
    Only weeks that can be int-cast are emitted.
    """
    out = {}
    for region, row in df.iterrows():
        series = {}
        for col in row.index:
            try:
                week = str(int(col))
            except Exception:
                continue
            series[week] = coerce_numeric(row[col])
        out[region] = series
    return out

def build_region_to_iso():
    """UN region -> list of ISO3 (via country_converter, like your notebook)"""
    try:
        import country_converter as coco
    except ImportError:
        raise SystemExit("Please install dependency: pip install country_converter")
    cc = coco.CountryConverter()
    mapping = {}
    df_cc = cc.data
    for region in UNRegion_list:
        iso_list = df_cc.loc[df_cc["UNregion"] == region, "ISO3"].dropna().astype(str).tolist()
        mapping[region] = iso_list
    return mapping

def extract_indoor_vital_per_person_from_main():
    """
    Use the main pickle to extract indoor-vital per-person bounds:
    first two columns = [lower_pp, upper_pp] (as in your notebook).
    """
    df = load_pickle_table(file_path_main)
    present = [r for r in UNRegion_list if r in df.index]
    if not present:
        raise KeyError(f"None of UNRegion_list found in {file_path_main}. "
                       f"Index sample: {list(df.index[:10])}")
    df = df.loc[present]
    indoor_vital_cols = df.columns[:2]
    indoor_vital = {}
    for r in present:
        row = df.loc[r, indoor_vital_cols]
        indoor_vital[r] = {
            "lower": coerce_numeric(row.iloc[0]),
            "upper": coerce_numeric(row.iloc[1]),
        }
    return indoor_vital

def main():
    print("[1/6] Loading datasets…")
    df_main,     wk_main    = nominal_df_from_pickle(file_path_main)
    df_percent,  wk_percent = nominal_df_from_pickle(file_path_percent)
    df_cr_man,   wk_man     = nominal_df_from_pickle(file_path_cr_man)
    df_cr_repur, wk_repur   = nominal_df_from_pickle(file_path_cr_repur)
    df_coalbag,  wk_coal    = nominal_df_from_pickle(file_path_coalbag)
    df_cr_stock, wk_stock   = nominal_df_from_pickle(file_path_cr_stock)
    df_portable, wk_port  = nominal_df_from_pickle(file_path_portable)

    # weeks = union length; JS will just use 1..max
    max_weeks = max(
        len(wk_main),
        len(wk_percent),
        len(wk_man),
        len(wk_repur),
        len(wk_coal),
        len(wk_stock),
        len(wk_port),
    )
    weeks = [str(i) for i in range(1, max_weeks + 1)]
    print(f"[2/6] Weeks detected: 1..{max_weeks}")

    print("[3/6] Building region_to_iso…")
    region_to_iso = build_region_to_iso()

    print("[4/6] Converting dataframes to JSON maps…")
    datasets = {
        "ALL":                  df_to_weekmap(df_main),
        "Indoor Vital Coverage %":       df_to_weekmap(df_percent),
        "CR Box Manufacturing": df_to_weekmap(df_cr_man),
        "CR Box Repurposing":   df_to_weekmap(df_cr_repur),
        "Coalbaghouse":         df_to_weekmap(df_coalbag),
        "CR Box Stock":         df_to_weekmap(df_cr_stock),
        "Portable Air Cleaners": df_to_weekmap(df_portable),
    }

    print("[5/6] Extracting indoor-vital per-person bounds (lower/upper) from main…")
    indoor_vital_per_person = extract_indoor_vital_per_person_from_main()

    bundle = {
        "weeks": weeks,
        "un_regions": UNRegion_list,
        "region_to_iso": region_to_iso,
        "datasets": datasets,
        "indoor_vital_per_person": indoor_vital_per_person,  # multiply by CADRPP client-side
    }

    OUTFILE.write_text(json.dumps(bundle))
    print(f"[6/6] Wrote {OUTFILE}")

if __name__ == "__main__":
    main()
