# In Room Air Filter Scale Up GitHub Repository
A scale up simulation of various in room filtration systems that could protect essential workers in the event of a catastrophic pandemic. 

---

## Overview
This git repository holds the code for the In Room Air Filtration ANU Capstone group in partnership with ALLFED. It contains the data processing and analysis of our simulations of scale up in room systems as well as our essential worker estimation given an extreme pandemic scenario. For more information, visit the ANU Capstone team's repositiory and landing page. 

Landing Page: https://sites.google.com/view/anu-capstone-air-filtration/home 

Repository: https://drive.google.com/drive/folders/1PC_QixM3_B3nh0tNhnJJnPxHECVEsJI7

---

## Project Structure
- [Data](#data)
- [Results](#results)
- [Scripts](#scripts)
- [SRC](#src)

---

## Installation
Clone the repository and install dependencies:

```
git clone https://github.com/SPROOK/InRoomAirFilterScaleUp.git 
cd InRoomAirFilterScaleUp
pip install -r requirements.txt
```

---

## Data
The `data` folder contains all source datasets used in our scale-up estimation and analysis. This includes data for our **Essential Worker Analysis**, **CR Box Analysis**, and **Coal Baghouse Analysis**.

### Essential Worker Analysis
- `ILO_ISCO_08_GLB.csv`  
- `Indoors_Environmentally_Controlled_data.csv`  
- `ISCO_SOC_Crosswalk.csv`  
- `ISCO-08 OpinionPollCensus.xlsx`  
- `LFData_WB_plus.xlsx`

### CR Box Analysis
- `CR_Box_Countries_MS.csv`

### Coal Baghouse Analysis
- `BaghouseAirflow.csv`

Additionally, the dataset `STANDARD_COUNTRY_LIST.csv` is used throughout the project to standardize the list of countries included in our scale-up estimation and simulations.

---

## Results
The `results` folder contains all outputs from our scale-up estimation and simulation processes.

### Output Images
Stored in a dedicated subfolder containing static visualizations and figures.

### Interactive Outputs
`.html` files demonstrate interactive simulations and visualizations generated as part of the analysis.

### Quantitative Results
Raw numerical results are stored in `.csv` files.  
When uncertainty values (`ufloats`) are included, a corresponding `.pkl` file is provided. These `.pkl` files preserve the uncertainty data structure so that results can be reloaded into code environments without loss of precision or data integrity.

---

## Scripts
The `scripts` folder contains all source code used to process data and generate results.  
It is organized into two main categories:

- **Processing Scripts** - handle data, transformation, and scale-up computations.  
- **Visualizer Scripts** - generate plots, charts, and interactive dashboards.

Each script is named to reflect the specific section of the analysis it supports and ends with a suffix


