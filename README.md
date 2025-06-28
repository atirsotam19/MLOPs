# Machine Learning Operations Project: Loan Approval Predictions  
## Loan Approval Prediction
### Master in Data Science and Advanced Analytics
---

### Team Members

- **Afonso Dias** – Student ID: 20211540
- **Leonor Mira** – Student ID: 20240658  
- **Martim Tavares** – Student ID: 20240508  
- **Rita Matos** – Student ID: 20211642
- **Rita Palma** – Student ID: 20240661


---
## Overview

This project is designed to implement MLOps practices for managing machine learning workflows using a [loan approval prediction dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset). The project is powered by Kedro and was generated using `kedro 0.19.12`.

## Rules and guidelines

In order to get the best out of the template:
- Don't remove any lines from the `.gitignore` file we provide.
- Don't commit data to your repository.
- Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`.

## Setup Instructions

### Step 1: Navigate to the Project Directory

Before running any commands, make sure you're in the root directory of the project. Use the `cd` command to change to your project folder:

```bash
cd "path_to_your_project"
```
### Step 2: Install Dependencies
To install the required Python packages listed in `requirements.txt`, run:
```bash
pip install -r requirements.txt
```

### Step 3: Run Pipelines
The `pipelines_registry.py` file registers several individual and composite pipelines, including:

- **Data cleaning and ingestion pipelines**

- **Data splitting and preprocessing pipelines**

- **Feature engineering and selection pipelines**

- **Model selection, training, and prediction pipelines**

- **Deployment and data drift monitoring pipelines**

It also defines two main composite pipelines:

**training_pipeline (default)**: combines all steps from data cleaning to model training.

**batch_inference_pipeline**: combines steps for batch predictions, deployment, and data drift detection.

## How to run your Kedro pipeline

These are the pipelines in pipeline_registry.py:

- Individual Pipelines
"data_cleaning": data_cleaning_pipeline.create_pipeline(),
"data_unit_tests": data_tests.create_pipeline(),
"data_ingestion": data_ingestion.create_pipeline(),
"data_split": split_data.create_pipeline(),
"data_preprocess_train": preprocess_train.create_pipeline(),
"data_preprocess_batch": preprocessing_batch.create_pipeline(),
"data_unit_tests_feature_engineering": data_unit_tests_feature_engineering.create_pipeline(),
"data_split_train": split_train.create_pipeline(),
"feature_selection": feature_selection_pipeline.create_pipeline(),
"model_selection": model_selection_pipeline.create_pipeline(),
"model_train": model_train_pipeline.create_pipeline(),
"model_predict": model_predict_pipeline.create_pipeline(),
"deployment": deployment_pipeline.create_pipeline(),
"data_drift": data_drift_pipeline.create_pipeline(),

- Group Pipelines
"training_pipeline": data_cleaning_pipeline + data_tests + data_ingestion + split_data + preprocess_train + preprocessing_batch + data_unit_tests_feature_engineering + split_train + feature_selection_pipeline + model_selection_pipeline + model_train_pipeline,
"batch_inference_pipeline": model_predict_pipeline + deployment_pipeline + data_drift_pipeline,
"__default__": training_pipeline,

You can run the default pipeline (training_pipeline) using the following command:
```bash
kedro run
```

To execute pipelines individually, run:
```bash
kedro run --pipeline pipeline_name
```

To execute groups of pipelines, run:
```bash
kedro run --pipeline pipeline_group_name
```

## Dataset
### Instructions
The `data/` folder has all types of data saved throughout the project. Place the raw dataset file in the `data/01_raw/` directory and then you'll be able to run everything and get the same results.

## Note: Every pipeline in src folder has a respective pytest in tests folder.

## How to use Pytest Coverage

For coverage testing with pytest:

```bash
pytest --cov=src
```