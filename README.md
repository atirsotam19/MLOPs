# Machine Learning Operations Project: Loan Approval Predictions

## Master in Data Science and Advanced Analytics

---

### Team Members

* **Afonso Dias** – 20211540
* **Leonor Mira** – 20240658
* **Martim Tavares** – 20240508
* **Rita Matos** – 20211642
* **Rita Palma** – 20240661

---

## Overview

This project applies MLOps principles to a **Loan Approval Prediction** task, using a publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset).

The project is built using the **Kedro** framework and was generated with version `0.19.12`. It features a modular and testable pipeline for end-to-end machine learning workflows, including model training, evaluation, and data drift monitoring.

---

## Project Guidelines

To maintain code quality and reproducibility, please follow these rules:

* Do **not** remove any lines from the provided `.gitignore`.
* Never commit data files to the repository.
* Never commit credentials or local configurations. Store these in `conf/local/`.

---

## Setup Instructions

### Navigate to the Project Directory

```bash
cd path_to_your_project
```

### Create a Virtual Environment and Install Dependencies

Create a virtual environment and then run:
```bash
pip install -r requirements.txt
```

---

## Running the Pipelines

### Registered Pipelines

The `pipeline_registry.py` file defines the following:

#### Individual Pipelines

* `data_cleaning`
* `data_unit_tests`
* `data_ingestion`
* `data_split`
* `data_preprocess_train`
* `data_preprocess_batch`
* `data_unit_tests_feature_engineering`
* `data_split_train`
* `feature_selection`
* `model_selection`
* `model_train`
* `model_predict`
* `deployment`
* `data_drift`

#### Group Pipelines

* **`training_pipeline`** *(default)*: Combines the full ML lifecycle — from data cleaning to model training.
* **`batch_inference_pipeline`**: Designed for batch predictions, deployment, and data drift detection.
* **`__default__`**: Alias for `training_pipeline`.

---

### Run the Pipelines

To run the **default training pipeline**:

```bash
kedro run
```

To run a **specific individual pipeline**:

```bash
kedro run --pipeline pipeline_name
```

To run a **grouped pipeline**:

```bash
kedro run --pipeline pipeline_group_name
```

---

## Dataset Usage

Place the raw dataset file inside the following directory:

```text
data/01_raw/
```

Once the raw data is in place, you can run the full pipeline to replicate results.

---

## Testing

Every pipeline module in the `src/` folder has a corresponding test module in the `tests/` folder.

### Run Tests with Coverage

To check code coverage using `pytest`:

```bash
pytest --cov=src
```
