"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from mlops_project.pipelines import (
    _00_data_cleaning as data_cleaning_pipeline,
    _01_data_unit_tests as data_tests,
    _02_data_ingestion as data_ingestion,
    _03_data_split as split_data,
    _04_preprocessing_train as preprocess_train,
    _05_preprocessing_batch as preprocessing_batch,
    _06_split_train_pipeline as split_train,
    _07_feature_selection as feature_selection_pipeline,
    _08_model_selection as model_selection_pipeline,
    _09_model_train as model_train_pipeline,
    _10_model_predict as model_predict_pipeline,
    _11_deployment as deployment_pipeline,
    _12_data_drift as data_drift_pipeline
)

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "data_cleaning": data_cleaning_pipeline.create_pipeline(),
        "data_unit_tests": data_tests.create_pipeline(),
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_split": split_data.create_pipeline(),
        "data_preprocess_train": preprocess_train.create_pipeline(),
        "data_preprocess_batch": preprocessing_batch.create_pipeline(),
        "data_split_train": split_train.create_pipeline(),
        "feature_selection": feature_selection_pipeline.create_pipeline(),
        "model_selection": model_selection_pipeline.create_pipeline(),
        "model_train": model_train_pipeline.create_pipeline(),
        "model_predict": model_predict_pipeline.create_pipeline(),
        "deployment": deployment_pipeline.create_pipeline(),
        "data_drift": data_drift_pipeline.create_pipeline(),

        # Optional combo pipelines
        "full_training_pipeline": data_cleaning_pipeline.create_pipeline() +
        data_tests.create_pipeline() +
        data_ingestion.create_pipeline() +
        split_data.create_pipeline() +
        preprocess_train.create_pipeline() +
        preprocessing_batch.create_pipeline() +
        split_train.create_pipeline() +
        feature_selection_pipeline.create_pipeline() +
        model_selection_pipeline.create_pipeline()  +
        model_train_pipeline.create_pipeline() +
        model_predict_pipeline.create_pipeline() +
        deployment_pipeline.create_pipeline() +
        data_drift_pipeline.create_pipeline()
    }
