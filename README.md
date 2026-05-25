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
| `results/visualizations/` | Interactive HTML choropleths |
| `src/preprocessing.py` | Load and transform raw essential-worker inputs |
| `src/essential_workers.py` | Overlap calibration, labour-force pipeline, validation |
| `src/countries.py` | CR-box / baghouse scale-up by country |
| `scripts/` | Processing notebooks (`essential_workers_processing`, `scale_up_processing`) |
| `scripts/visualization/` | Plotly choropleths and scale-up visualisers |

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

### Scale-up analysis (`data/scale_up/`)

- `STANDARD_COUNTRY_LIST.csv`
- `CR_Box_Countries_MS.csv`
- `BaghouseAirflow.csv`

---

## Results

- **`results/essential_workers/`** — per-country/regional worker counts, validation, overlap calibration, on-site housing requirements
- **`results/scale_up/`** — CADR trajectories (`.csv` / `.pkl`), time-to-reach tables
- **`results/visualizations/`** — `.html` choropleth maps

---

## Scripts

**Processing**

- `scripts/essential_workers_processing.ipynb` — walkthrough of the essential-worker pipeline (including indoor-fraction method comparison)
- `scripts/scale_up_processing.ipynb` — CR-box / baghouse scale-up (requires essential-worker outputs)

**Visualization** (`scripts/visualization/`)

- `EssentialWorkers_Choropleth_Visualiser.ipynb`
- `Scale_Up_Visualiser.ipynb`
- `Airflow_Visualiser.ipynb`
- `Time_To_Cover_Choropleth_Visualiser.ipynb`
- `UNRegion_Choropleth_Visualiser.py` (helper module)

---

## Running pipelines from the command line

```bash
PYTHONPATH=src python -m essential_workers
PYTHONPATH=src python -c "from countries import run_pipeline; from paths import SCALE_UP_DATA, SCALE_UP_RESULTS; run_pipeline(SCALE_UP_DATA, SCALE_UP_RESULTS, write=True)"
```

## Tests

```bash
pytest                    # fast, uses tests/fixtures/
pytest --full-data        # includes ILO sense-checks on real data
```
