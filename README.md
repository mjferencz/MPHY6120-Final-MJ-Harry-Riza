**MPHY 6120 - Final: Predicting Cancer Lineage and Subtype with MLP**

## Overview
- PLEASE NOTE: Running this code was designed around CPU usage and was not tested for GPU. On CPU, it will take 5-10 minutes to run depending on CPU type, so please be patient.
- The purpose and use of this model is explained in detail in Final_Project_Field_Guide_V2.pdf. For a brief overview, make sure packages listed in requirements.txt and README.md are downloaded and up to date. Make sure files in data/ are properly downloaded (especially data/crispr_data.csv as it is the largest and prone to duplication error). This code will return a trained model to evaluate lineage and subtype simply by runnning. To use further, you can modify to output the model and add a new file into data/ containing a single patient's CRISPR Dependency Scores. Call for an explicit report of most probable lineages and subtypes based on these scores which should also contain model confidence. Take into account these relative confidence factors before preceding with any decisions. See the aforementioned field guide for clinical implementation steps.
- Data loading and processing with imputation and insufficient class size exclusion.
- EDA: Sample gene correlations, distributions of dependency scores, class imbalance visualizations, correlations between genes and lineages/subtypes
- Preparation of data splitting with and without SMOTE.
- MLP neural network architecture using Plateau Scheduler, CrossEntropyLoss, Adam optimizer.
- Using training and validation data on the model.
- Using testing data on MLP and baseline (most_frequent strategy).
- Evaluating with Accuracy, Top 3 Accuracy, F1-Macro, F1-Weighted, AUC-ROC, Matthews Correlation Coefficient.
- Using SHAP to see most important features (genes).

## Set-Up
```bash
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install pandas numpy matplotlib seaborn sklearn torch imblearn shap
```

## Datasets
- CRISPR Dependency Scores (data/crispr_data.csv): Dependency of cancer samples on 500 genes, testing via CRISPR KO & remaining viability
- Cancer Lineage (data/lineage_data.csv): Lineage of each cancer
- Cancer Subtype (data/subtype_data.csv): Subtype of each cancer (more specific classification within lineage)
