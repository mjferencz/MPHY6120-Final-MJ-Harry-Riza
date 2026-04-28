**Dataset Information**

## Description
- CRISPR Dependency Scores (data/crispr_data.csv): Dependency of cancer samples on 500 genes, testing via CRISPR KO & remaining viability
- Cancer Lineage (data/lineage_data.csv): Lineage of each cancer
- Cancer Subtype (data/subtype_data.csv): Subtype of each cancer (more specific classification within lineage)

## Notes
- Do not try to load data/crispr_data.csv via Explorer, file is too large to load in VS Code and may cause future issues. If you wish to view, download to local drive first. data/lineage_data.csv and data/subtype_data.csv are fine to preview in VS Code.
- Not all patients are the same between the two, IDs are compared within the main code to match between datasets (preferred to keep the raw data in data/).