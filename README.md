# FDS_COMP_2025 — Kaggle Competition (Pokémon Battle Prediction 2025)

This repository contains the material developed for the Kaggle competition **“FDS — Pokémon Battles Prediction 2025,”** as part of the Fundamentals of Data Science course at Sapienza University of Rome.

## Repository Contents
```text
FDS_COMP_2025/
│
├── main.py                          # Main notebook (orchestrates all three models)
│
├── model-1.py                       # Model 1: Python module for “fork-of-vit-notebook-fds2-final.py”
├── model-2.py                       # Model 2: Python module for “fork-of-vit-notebook-fds.py”
├── model-3.py                       # Model 3: Python module for “mrk-notebook-fds.py”
│
└── README.md                        # Project documentation
```
**Note**: Each *model* file is derived from the corresponding notebook listed above. Since each notebook has its own feature engineering, feature selection and modeling steps, the models do not share common functions. For this reason, creating shared reusable modules would not be meaningful, so they were not included.

## How to use 
To run the project, simply execute the `main.py` file in Kaggle or VS Code.
The script imports the three model modules and runs them to generate the submission files.

## Short description of the models
### Model 1 — “fork-of-vit-notebook-fds2-final.py”
A model with an extensive feature engineering pipeline, feature selection via correlation-based filters and L1 regularization and a multi-model stacking ensemble with optimized decision thresholding.

### Model 2 — “fork-of-vit-notebook-fds.py”
A model with a structured feature engineering pipeline, feature selection via correlations and Top-K ranking and a calibrated multi-model stacking ensemble with optimized decision thresholding.

### Model 3 — “mrk-notebook-fds.py”
A model with a targeted feature engineering pipeline, feature selection via VIF pruning and Elastic Net and a multi-model stacking ensemble with multi-seed validation and optimized decision thresholding.

## Authors

| Name | Student ID | Email |
|------|------------|-------|
| **Vitaliano Barberio** | *1992511* | *barberio.1992511@studenti.uniroma1.it* |
| **Debora Siri** | 1921846 | *siri.1921846@studenti.uniroma1.it* |
| **Mirko Impera** | *2254488* | *impera.2254488@studenti.uniroma1.it* |
