
# MCI-to-AD Conversion Prediction on ADNI

## 1. Project Overview

This project investigates the prediction of conversion from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) using longitudinal and multimodal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

The primary task is binary classification of:
- Progressive MCI (pMCI)
- Stable MCI (sMCI)

The prediction horizon is 36 months from baseline.

## 2. Dataset

- Source: Alzheimer's Disease Neuroimaging Initiative (ADNI).
- Main dataset: ADNIMERGE.
- Data type: Longitudinal tabular data.
- Observation unit: Individual participant (RID).
- Main input groups: Demographic, genetic, cognitive, MRI-derived, CSF and PET features.

Raw ADNI data are stored locally and are not included in this repository.

## 3. Repository Structure

- `src/cohort.py`: MCI cohort construction and validation.
- `src/datacard.py`: Dataset statistics and data quality summaries.
- `notebooks/01_datacard.ipynb`: Initial dataset inspection and Data Card generation.
- `reports/`: Aggregated statistics and research summaries.
- `data/`: Local data directory (excluded from Git).

## 4. Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Data Preparation

1. Obtain authorized access to ADNI.
2. Download ADNIMERGE and its data dictionary.
3. Place the dataset in `data/raw/`.
4. Record the dataset version and download date.
5. Run `notebooks/01_datacard.ipynb`.

## 6. Research Scope

The MVP focuses on:
- MCI cohort construction.
- Binary classification at a 36-month horizon.
- Initial dataset inspection and Data Card.
- Missing data and modality availability analysis.

Survival analysis is considered an extension and is not part of the initial MVP.

## 7. Data Protection

Raw ADNI data, CSV files, R data files and downloaded archives must not be committed to this repository.

## 8. Reproducibility

Record Python and library versions, dataset version, processing configuration and random seeds when applicable.
