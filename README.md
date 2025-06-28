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

This project is designed to implement MLOps practices for managing machine learning workflows. The project is powered by Kedro and was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:
- Don't remove any lines from the `.gitignore` file we provide.
- Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention).
- Don't commit data to your repository.
- Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- `pip` (Python package manager)
- Make (optional, for running `Makefile` commands)

## Setup Instructions

### Step 1: Navigate to the Project Directory

Before running any commands, make sure you're in the root directory of the project. Use the `cd` command to change to your project folder:

```bash
cd "path_to_your_project"
```
### Step 2: Install Dependencies
To install the required Python packages listed in `requirements.txt`, run:
```bash
make install
```
Alternatively, if you don't have Make installed, you can manually run:
```bash
pip install -r requirements.txt
```

### Step 3: Run Pipelines
To execute the pipelines defined in `pipelines_registry.py`, run:
```bash
make run_pipelines
```
Or manually:
```bash
python pipelines_registry.py
```
The `pipelines_registry.py` file registers several individual and composite pipelines, including:

- **Data cleaning and ingestion pipelines**

- **Data splitting and preprocessing pipelines**

- **Feature engineering and selection pipelines**

- **Model selection, training, and prediction pipelines**

- **Deployment and data drift monitoring pipelines**

It also defines two main composite pipelines:

**training_pipeline**: combines all steps from data cleaning to model training.

**batch_inference_pipeline**: combines steps for batch predictions, deployment, and data drift detection.

## How to run your Kedro pipeline

You can run your Kedro project with:
```bash
kedro run
```
## Dataset
### Location
The dataset for this project is located in the `data/` directory. Ensure that the dataset files are placed in the appropriate subdirectories:
- **`data/01_raw/`**: Contains raw, unprocessed data directly sourced from external systems.
- **`data/02_intermediate/`**: Contains data after initial preprocessing steps, such as cleaning or transformation.
- **`data/03_primary/`**: Contains primary datasets that are ready for feature engineering.
- **`data/04_feature/`**: Contains datasets with engineered features for modeling.
- **`data/05_model_input/`**: Contains datasets prepared for model training and evaluation.
- **`data/06_models/`**: Contains trained models and their metadata.
- **`data/07_model_output/`**: Contains data used for inference or predictions with trained models.
- **`data/08_reporting/`**: Contains reporting data, such as metrics, visualizations, or summaries.

### Instructions
1. Place the raw dataset files in the `data/01_raw/` directory.
2. Run the preprocessing pipeline to generate intermediate and primary datasets:
   ```bash
   kedro run --pipeline preprocess_data
   ```
3. Run the feature engineering pipeline to generate feature datasets:
   ```bash
   kedro run --pipeline feature_engineering
   ```
4. Use the `data/05_model_input/` directory for training and evaluation datasets.
5. Trained models will be saved in the `data/06_models/` directory.
6. Use the `data/07_model_output/` directory for inference datasets.
7. Reporting outputs will be saved in the `data/08_reporting/` directory.

## Note: Every pipeline in src folder has a respective pytest in tests folder.

## How to test your Kedro project

Have a look at the files `tests/pipelines/` for instructions on how to write your tests. Run the tests as follows:
```bash
pytest
```
To run the entire project pipelines:

```bash
kedro run
```

To run only the batch inference pipeline:

```bash
kedro run --pipeline batch_inference_pipeline
```
For coverage testing with pytest:

```bash
pytest --cov=src
```

To configure the coverage threshold, look at the `.coveragerc` file.

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.



### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with:
```bash
nbstripout --install
```
This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.


## File Structure
- **`requirements.txt`**: Contains the list of Python dependencies for the project.
- **`pipelines_registry.py`**: The main script for running pipelines.
