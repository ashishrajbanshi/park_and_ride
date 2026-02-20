# park_and_ride

# prerequisite
create a virtual python environment
```CLI
python -m venv .venv
pip install pandas geopandas matplotlib pyyaml scikit-learn
```

## Automating the project
```CLI
sh automate.sh #only runs in linux environment
bat automate.bat #only runs in windows environment
```

## Structure
configuration is done in config folder. Weights, coordinates system are configured in config.yml file

- Preparation data: Data mapping, preparing census blocks, parking lots preparation are done in data preparation.
- Transit Accessibility: Accessibility of sites are calculated based on the each weights of transit feature.
- Clustering: Removing the clustered lots using DBSCAN
- Equity Calculations: Spatial equity is calculated here (This is no longer needed)
- Optimization: Final selection of sites are done in this step.

## Sensitivity Analysis

Sensitivity analysis of weights are done using weight_sensitivity_analysis.py
Complete sensitivity analysis:
1. Accessibility component weights (Level 1)
2. Final score composition weights (Level 2: Access + Equity + VW_Distance)