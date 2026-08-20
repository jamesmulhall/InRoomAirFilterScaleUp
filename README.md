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
| `data/scale_up/` | Country list, CR box parameters, coal baghouse airflow |
| `results/essential_workers/` | Essential/vital worker CSV outputs |
| `results/scale_up/` | Scale-up trajectory CSV/PKL and time-to-reach tables |
| `results/visualizations/` | HTML choropleths and ALLFED-styled PNG maps |
| `src/preprocessing.py` | Load and transform raw essential-worker inputs |
| `src/essential_workers.py` | Overlap calibration, labour-force pipeline, validation |
| `src/countries.py` | CR-box / baghouse / commercial-filter scale-up by country |
| `scripts/` | Processing notebooks (`essential_workers_processing`, `scale_up_processing`) |
| `scripts/visualization/` | Plotly choropleths, ALLFED matplotlib maps, scale-up visualisers |

---

## Installation

```bash
git clone https://github.com/SPROOK/InRoomAirFilterScaleUp.git
cd InRoomAirFilterScaleUp
pip install -r requirements.txt
pip install -e .
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

- `STANDARD_COUNTRY_LIST.csv`
- `CR_Box_Countries_MS.csv`
- `BaghouseAirflow.csv`

---

## Results

- **`results/essential_workers/`** — per-country/regional worker counts, per-group worker counts and ASHRAE-241 CADR requirements (ECA × 5.7), validation, overlap calibration, on-site housing requirements
- **`results/scale_up/`** — CADR trajectories (`.csv` / `.pkl`), time-to-reach tables
- **`results/visualizations/`** — `.html` choropleth maps and `.png` ALLFED-styled figures

---

## Scripts

**Processing**

- `scripts/essential_workers_processing.ipynb` — walkthrough of the essential-worker pipeline (including indoor-fraction method comparison)
- `scripts/scale_up_processing.ipynb` — CR-box / baghouse / commercial-filter scale-up (requires essential-worker outputs)

**Visualization** (`scripts/visualization/`)

- `EssentialWorkers_Choropleth_Visualiser.ipynb`
- `Scale_Up_Visualiser.ipynb`
- `Airflow_Visualiser.ipynb`
- `Time_To_Cover_Choropleth_Visualiser.ipynb`
- `UNRegion_Choropleth_Visualiser.py` (helper module)
- `plot_essential_workers.py` — ALLFED-styled world maps of essential/vital worker shares (requires `geopandas`, install via `mamba install -c conda-forge geopandas`)
- `plot_filtration_coverage.py` — filtration coverage maps at week 12 (by country and by UN region); total CADR line plots, stacked area plots, and stacked line plots by source, each with the vital–essential CADR band (same deps; 300 DPI PNGs; fetches ALLFED style sheet from GitHub on first run)

---

## Running pipelines from the command line

```bash
PYTHONPATH=src python src/essential_workers.py
PYTHONPATH=src python src/countries.py
conda run -n InRoomAirFilterScaleUp python scripts/visualization/plot_essential_workers.py
conda run -n InRoomAirFilterScaleUp python scripts/visualization/plot_filtration_coverage.py
```

## Tests

```bash
pytest                    # fast, uses tests/fixtures/
pytest --full-data        # includes ILO sense-checks on real data
```
