# In Room Air Filter Scale Up GitHub Repository

A scale up simulation of various in room filtration systems that could protect essential workers in the event of a catastrophic pandemic.

---

## Overview

This repository holds the data processing and analysis for in-room filtration scale-up simulations and essential-worker estimation under an extreme pandemic scenario.

Landing Page: https://sites.google.com/view/anu-capstone-air-filtration/home

Repository: https://drive.google.com/drive/folders/1PC_QixM3_B3nh0tNhnJJnPxHECVEsJI7

---

## Project structure

| Path | Purpose |
| --- | --- |
| `data/essential_workers/` | ILO, O*NET, poll, crosswalk, labour force, JEM (optional) |
| `data/scale_up/` | Parameters, settings, coal baghouse airflow, cached World Bank MVA |
| `results/essential_workers/` | Essential/vital worker CSV outputs |
| `results/scale_up/` | Weekly eCADR and workforce coverage for each scenario |
| `results/visualizations/` | ALLFED-styled manuscript figures (300 DPI PNG) |
| `src/preprocessing.py` | Load and transform raw essential-worker inputs |
| `src/essential_workers.py` | Overlap calibration, labour-force pipeline, validation |
| `src/scale_up_model.py` | Methods 2.3 scale-up: PACs, CR boxes, coal baghouse filters |
| `src/linear_models.py` | Fits the coal-airflow and MVA-exponent regressions the model uses |
| `src/mc_distributions.py` | Monte Carlo samplers for the uncertain parameters |
| `scripts/` | Processing notebooks (`essential_workers_processing`, `scale_up_processing`) |
| `scripts/visualization/` | ALLFED matplotlib figure scripts |

---

## Installation

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
git clone https://github.com/SPROOK/InRoomAirFilterScaleUp.git
cd InRoomAirFilterScaleUp
uv sync --extra dev --extra notebooks
```

That creates `.venv`, installs the pinned dependencies and installs this
repository in editable mode, so `import scale_up_model` works from anywhere.

Without uv:

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is generated from the lockfile, so regenerate it after
changing dependencies in `pyproject.toml`:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

---

## Data

### Essential worker analysis (`data/essential_workers/`)

- `ILO_ISCO_08_GLB.csv`
- `Indoors_Environmentally_Controlled_data.csv`
- `Indoors_Not_Environmentally_Controlled.csv`
- `job_exposure_matrix.xls` (optional; JEM indoor sensitivity)
- `ISCO_SOC_Crosswalk.csv`
- `ISCO-08 OpinionPollCensus.xlsx`
- `LFData_WB_plus.xlsx`
- `ILO_country_essential_workers_pct.xlsx`
- `ASHRAE241_ECA_by_occupancy.csv` — ASHRAE 241 Table 5-1 ECA rates (from Jones et al. 2025 Table 4)
- `ASHRAE241_group_mapping.csv` — maps occupational groups to ASHRAE occupancy categories

### Scale-up analysis (`data/scale_up/`)

- `parameters.csv` — uncertain parameters (low, high, distribution, units, note, source)
- `settings.csv` — fixed settings, including the width of the reported uncertainty interval, `adjust_MVA_by_cost`, and `linear_fit_PRODCOM_only`; `linear_models.py` writes its fitted values into this file
- `allocator_fit_data.csv` — the 40 observations behind the MVA exponent
- `coal_plant_airflow.csv` — coal plant capacity and baghouse airflow sample
- `BaghouseAirflow.csv` — coal operating MW per country
- `mva_world_bank.csv` — cached manufacturing value added, downloaded on first run
- `comtrade_HS842139.xlsx` — UN Comtrade HS 842139 export value and net weight, used when `adjust_MVA_by_cost` is on

---

## Results

- **`results/essential_workers/`** — per-country/regional worker counts, per-group worker counts and ASHRAE-241 CADR requirements (ECA × 5.7), validation, overlap calibration, on-site housing requirements
- **`results/scale_up/`** — for each scenario: `weekly_ecadr_by_country_*`, `ecadr_by_channel_*` (weekly, global) and `coverage_{vital,essential}_*` (median and uncertainty interval by region and week), plus `requirements_by_region.csv`, the eCADR each region is measured against
- **`results/linear_models/`** — plots of the two fitted regressions
- **`results/visualizations/`** — ALLFED-styled manuscript figures

---

## Scripts

**Processing**

- `scripts/essential_workers_processing.ipynb` — walkthrough of the essential-worker pipeline (including indoor-fraction method comparison)
- `scripts/scale_up_processing.ipynb` — walkthrough of the scale-up model, stage by stage, ending in the three manuscript figures (requires the essential-worker outputs)

**Visualization** (`scripts/visualization/`)

Every script writes 300 DPI PNGs to `results/visualizations/` and fetches the
ALLFED style sheet and Natural Earth country polygons from the internet on first
run. Maps use the Winkel Tripel projection. Colormaps come from
[cmasher](https://cmasher.readthedocs.io/) via `get_cmap()` in `viz_common.py`.
Set `SHOW_MAP_BORDER = True` in that file to draw the ALLFED outline on every
choropleth. `CMAP_RANGE` in `plot_filtration_coverage.py` keeps only part of
the named colormap (the default drops the black tip of `arctic_r`).

- `plot_essential_workers.py` — `PctVitalWorkers_Manuscript`, `PctEssentialWorkers_Manuscript` and the four-panel `PctWorkersByCountry_Manuscript_2x2`
- `plot_workers_vs_gdp.py` — `WorkerShares_vs_GDP_PPP` and `FoodShareOfWorkforce_vs_GDP_PPP`, and the correlation tables behind them
- `plot_group_composition.py` — `GroupComposition_Global`, the occupational make-up of the workforces
- `plot_filtration_coverage.py` — `ScenarioCoverage_Manuscript` (all three scenarios with uncertainty intervals), `Global_stacked_cadr` (supply by channel), `FiltrationCoverageByRegion_Manuscript_Week13` (two-panel essential and vital maps), `FiltrationSupplyAndCoverage_Manuscript_Week13` (regional eCADR supply above, vital coverage below), plus single-panel vital and essential maps. Coverage is a share of the indoor vital requirement. Pass `--scenario` for the stacked figure and the maps, and `--week` for the mapped week

---

## Running pipelines from the command line

The scale-up model needs the essential-worker outputs, and reads the fitted
values from `settings.csv`, so run them in this order:

```bash
python src/essential_workers.py
python src/linear_models.py    # fits the regressions and updates settings.csv
python src/scale_up_model.py   # all three scenarios
python scripts/visualization/plot_essential_workers.py
python scripts/visualization/plot_workers_vs_gdp.py
python scripts/visualization/plot_group_composition.py
python scripts/visualization/plot_filtration_coverage.py
```

`linear_models.py` writes `baghouse_gradient`, `baghouse_intercept_l_per_s` and
`mva_exponent_b` straight into `data/scale_up/settings.csv`, each with the R² and
sample size it came from, so there is nothing to copy by hand. Set
`linear_fit_PRODCOM_only` to `1` to fit `b` on PRODCOM sold production only.
Its plots go to `results/linear_models/`.

Both model scripts refuse to run on incomplete inputs: `linear_models.py` stops
while `coal_plant_airflow.csv` is empty, and `scale_up_model.py` stops while any
parameter in `parameters.csv` still has `low == high == 0`, naming each one. On
the first run `scale_up_model.py` downloads manufacturing value added from the
World Bank; check the printed country count is roughly 190-200, since a much
lower number means the download failed and a stale cache was used.

## Tests

```bash
pytest                    # fast, uses tests/fixtures/
pytest --full-data        # includes ILO sense-checks on real data
```
