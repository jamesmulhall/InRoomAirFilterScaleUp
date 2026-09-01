"""
Fit the two linear models that feed the in-room filtration scale-up model.

Both were previously fitted outside the repository in a spreadsheet. They are
here so the whole calculation lives in one place:

  1. Coal plant capacity (MW) to baghouse airflow (L/s), from a small sample of
     plants. Used by methods equation 7.
  2. Filtration output against manufacturing value added. By default this pools
     country-level filtration market revenue with Eurostat PRODCOM sold
     production. If ``linear_fit_PRODCOM_only`` is set in settings.csv, only
     the PRODCOM points are used. The slope is the exponent b in methods
     equation 3.

Running this script writes one plot per model and updates the three fitted rows
in data/scale_up/settings.csv, which is where the model reads them from.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set up ALLFED plotting style
plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

COAL_FILE = "data/scale_up/coal_plant_airflow.csv"
ALLOCATOR_FILE = "data/scale_up/allocator_fit_data.csv"
SETTINGS_FILE = "data/scale_up/settings.csv"
RESULTS_DIR = "results/linear_models"

# Dataset labels used in the allocator input file
DATASETS = {
    "GrandView": "Filtration market revenue (world)",
    "PRODCOM": "PRODCOM 28251410 (EU production)",
}


def fit_coal_airflow(path=COAL_FILE):
    """
    Fit baghouse airflow against coal plant capacity.

    Arguments:
        path (str): CSV with columns plant, capacity_mw, airflow_l_per_s.

    Returns:
        dict: slope, intercept, r_squared and the sample size.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"{path} is empty. Add the coal plant sample from the methods "
            "Supplementary Information (columns: plant, capacity_mw, airflow_l_per_s)."
        )

    model = sm.OLS(
        df.airflow_l_per_s.to_numpy(float),
        sm.add_constant(df.capacity_mw.to_numpy(float)),
    ).fit()
    params = np.asarray(model.params)
    return {
        "slope": params[1],
        "intercept": params[0],
        "r_squared": model.rsquared,
        "n": int(model.nobs),
        "data": df,
    }


def plot_coal_airflow(fit, path):
    """
    Plot the coal capacity to airflow fit.

    Arguments:
        fit (dict): Output of fit_coal_airflow.
        path (str): Where to save the figure.
    """
    df = fit["data"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df.capacity_mw, df.airflow_l_per_s, zorder=3)
    x = np.linspace(0, df.capacity_mw.max() * 1.05, 50)
    ax.plot(x, fit["intercept"] + fit["slope"] * x, zorder=2)
    ax.set_xlabel("Coal plant capacity (MW)")
    ax.set_ylabel("Baghouse airflow (L/s)")
    ax.set_title(
        f"Airflow = {fit['slope']:.0f} × MW + {fit['intercept']:,.0f}\n"
        f"R² = {fit['r_squared']:.2f}, n = {fit['n']}",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_allocator(path=ALLOCATOR_FILE, prodcom_only=False):
    """
    Fit filtration output against MVA.

    The default pooled model shares one slope across both datasets and gives
    each its own intercept, so the slope is not distorted by the difference
    in levels. If ``prodcom_only`` is True, that pooled regression is skipped
    and the slope comes from the PRODCOM points alone.

    Arguments:
        path (str): CSV with columns country, dataset, value_usd, mva_usd.
        prodcom_only (bool): Fit b on PRODCOM only.

    Returns:
        dict: Single-dataset fits, the input data, and either the pooled or
            the PRODCOM model as ``chosen``.
    """
    df = pd.read_csv(path)
    df["log_y"] = np.log10(df.value_usd)
    df["log_x"] = np.log10(df.mva_usd)
    df["is_prodcom"] = (df.dataset == "PRODCOM").astype(float)

    single = {}
    for name in DATASETS:
        subset = df[df.dataset == name]
        single[name] = sm.OLS(
            subset.log_y.to_numpy(), sm.add_constant(subset.log_x.to_numpy())
        ).fit()

    if prodcom_only:
        return {
            "pooled": None,
            "single": single,
            "chosen": single["PRODCOM"],
            "prodcom_only": True,
            "data": df,
        }

    pooled = sm.OLS(
        df.log_y.to_numpy(),
        sm.add_constant(np.column_stack([df.log_x, df.is_prodcom])),
    ).fit()
    return {
        "pooled": pooled,
        "single": single,
        "chosen": pooled,
        "prodcom_only": False,
        "data": df,
    }


def plot_allocator(fit, path):
    """
    Plot the fit that supplies b, with the two single-dataset fits beside it.

    Arguments:
        fit (dict): Output of fit_allocator.
        path (str): Where to save the figure.
    """
    df = fit["data"]
    fig = plt.figure(figsize=(12.5, 5.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.6, wspace=0.25)
    main = fig.add_subplot(grid[:, 0])
    panels = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]

    x = np.linspace(df.log_x.min() - 0.12, df.log_x.max() + 0.12, 50)
    chosen = fit["chosen"]
    if fit["prodcom_only"]:
        subset = df[df.dataset == "PRODCOM"]
        intercept, slope = np.asarray(chosen.params)
        main.scatter(subset.log_x, subset.log_y, label=DATASETS["PRODCOM"], zorder=3)
        main.plot(x, intercept + slope * x, zorder=2)
        main.annotate(
            f"log$_{{10}}$y = {intercept:.2f} + {slope:.3f}·log$_{{10}}$MVA\n"
            f"b = {slope:.3f} ± {np.asarray(chosen.bse)[1]:.3f}\n"
            f"R² = {chosen.rsquared:.3f}",
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            va="bottom",
            ha="right",
        )
        main.set_title(
            f"PRODCOM-only fit (n = {int(chosen.nobs)})",
            loc="left",
        )
    else:
        intercept, slope, shift = np.asarray(chosen.params)
        for name, label in DATASETS.items():
            subset = df[df.dataset == name]
            main.scatter(subset.log_x, subset.log_y, label=label, zorder=3)
            main.plot(x, intercept + shift * (name == "PRODCOM") + slope * x, zorder=2)
        sign = "+" if shift >= 0 else "-"
        main.annotate(
            f"log$_{{10}}$y = {intercept:.2f} {sign} {abs(shift):.2f}·PRODCOM "
            f"+ {slope:.3f}·log$_{{10}}$MVA\n"
            f"b = {slope:.3f} ± {np.asarray(chosen.bse)[1]:.3f}\n"
            f"R² = {chosen.rsquared:.3f}",
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            va="bottom",
            ha="right",
        )
        main.set_title(
            "Pooled fit — shared slope, dataset-specific intercept "
            f"(n = {int(chosen.nobs)})",
            loc="left",
        )

    main.set_xlabel("log$_{10}$ manufacturing value added (USD)")
    main.set_ylabel("log$_{10}$ annual value (USD)")
    main.legend(loc="upper left")

    for axis, (name, label) in zip(panels, DATASETS.items()):
        subset = df[df.dataset == name]
        model = fit["single"][name]
        coeffs = np.asarray(model.params)
        xi = np.linspace(subset.log_x.min() - 0.1, subset.log_x.max() + 0.1, 50)
        axis.scatter(subset.log_x, subset.log_y, zorder=3)
        axis.plot(xi, coeffs[0] + coeffs[1] * xi, zorder=2)
        axis.annotate(
            f"log$_{{10}}$y = {coeffs[0]:.2f} + {coeffs[1]:.3f}·log$_{{10}}$MVA\n"
            f"b = {coeffs[1]:.3f} ± {np.asarray(model.bse)[1]:.3f}\n"
            f"R² = {model.rsquared:.3f}",
            xy=(0.96, 0.05),
            xycoords="axes fraction",
            va="bottom",
            ha="right",
            fontsize=8,
        )
        axis.set_title(f"{label} (n = {int(model.nobs)})", loc="left", fontsize=9)
        axis.set_ylabel("log$_{10}$ value", fontsize=9)
    panels[1].set_xlabel("log$_{10}$ manufacturing value added (USD)", fontsize=9)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_settings(coal, allocator, path=SETTINGS_FILE):
    """
    Write the fitted values into the settings table the model reads.

    The three fitted rows are replaced where they already sit, so
    data/scale_up/settings.csv stays the only place the model's fixed settings
    live and the fits cannot go stale against it.

    Arguments:
        coal (dict): Output of fit_coal_airflow.
        allocator (dict): Output of fit_allocator.
        path (str): Path to the settings CSV.

    Returns:
        pandas.DataFrame: The rows that were written.
    """
    chosen = allocator["chosen"]
    if allocator["prodcom_only"]:
        b_note = (
            f"Fitted in linear_models.py on PRODCOM only. Standard error "
            f"{np.asarray(chosen.bse)[1]:.3f}, R2 = {chosen.rsquared:.3f}, "
            f"n = {int(chosen.nobs)}."
        )
    else:
        b_note = (
            f"Fitted in linear_models.py. Standard error "
            f"{np.asarray(chosen.bse)[1]:.3f}, R2 = {chosen.rsquared:.3f}, "
            f"n = {int(chosen.nobs)}."
        )
    rows = [
        {
            "setting": "baghouse_gradient",
            "value": round(coal["slope"], 2),
            "units": "L/s per MW",
            "note": "Fitted in linear_models.py. "
            f"R2 = {coal['r_squared']:.2f}, n = {coal['n']}.",
            "source": COAL_FILE,
        },
        {
            "setting": "baghouse_intercept_l_per_s",
            "value": round(coal["intercept"], 1),
            "units": "L/s",
            "note": "Fitted in linear_models.py. Per-plant intercept, so it is only "
            "applied to countries with non-zero coal capacity.",
            "source": COAL_FILE,
        },
        {
            "setting": "mva_exponent_b",
            "value": round(np.asarray(chosen.params)[1], 3),
            "units": "exponent",
            "note": b_note,
            "source": ALLOCATOR_FILE,
        },
    ]

    settings = pd.read_csv(path).set_index("setting")
    for row in rows:
        columns = [key for key in row if key != "setting"]
        settings.loc[row["setting"], columns] = [row[key] for key in columns]
    settings.reset_index().to_csv(path, index=False)
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("LINEAR MODELS FEEDING THE SCALE-UP MODEL")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nFitting coal capacity to baghouse airflow...")
    coal = fit_coal_airflow()
    print(
        f"  airflow = {coal['slope']:.0f} x MW + {coal['intercept']:,.0f}, "
        f"R2 = {coal['r_squared']:.2f}, n = {coal['n']}"
    )
    plot_coal_airflow(coal, os.path.join(RESULTS_DIR, "coal_airflow_fit.png"))

    print("\nFitting filtration output against MVA...")
    from scale_up_model import load_settings

    prodcom_only = bool(load_settings().get("linear_fit_PRODCOM_only"))
    allocator = fit_allocator(prodcom_only=prodcom_only)
    chosen = allocator["chosen"]
    label = "PRODCOM-only" if prodcom_only else "pooled"
    print(
        f"  {label} b = {np.asarray(chosen.params)[1]:.3f} "
        f"(SE {np.asarray(chosen.bse)[1]:.3f}), R2 = {chosen.rsquared:.3f}, "
        f"n = {int(chosen.nobs)}"
    )
    for name, model in allocator["single"].items():
        print(
            f"  {name:10s} b = {np.asarray(model.params)[1]:.3f}, "
            f"R2 = {model.rsquared:.3f}, n = {int(model.nobs)}"
        )
    plot_allocator(allocator, os.path.join(RESULTS_DIR, "mva_allocator_fit.png"))

    fitted = update_settings(coal, allocator)
    print(f"\nPlots written to {RESULTS_DIR}/")
    print(f"Fitted values written into {SETTINGS_FILE}:")
    for row in fitted.itertuples():
        print(f"  {row.setting} = {row.value}")


if __name__ == "__main__":
    main()
