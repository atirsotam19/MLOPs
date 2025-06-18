"""
This is a boilerplate pipeline 'preprocessing_train'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import treat_outliers, feature_engineer, scale_encode


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=treat_outliers,
            inputs="raw_train_data", ## MUDAR
            outputs="outlier_treated_data",
            name="treat_outliers_node"
        ),
        node(
            func=feature_engineer,
            inputs="outlier_treated_data",
            outputs="feature_created_data",
            name="feature_engineering_node"
        ),
        node(
            func=scale_encode,
            inputs="feature_created_data",
            outputs=[
                "preprocessed_train_data",
                "trained_standard_scaler",
                "trained_one_hot_encoder"
            ],
            name="scale_and_encode_node"
        )
    ])


# preprocessed_train_data:
#   type: pandas.CSVDataset
#   filepath: data/03_primary/preprocessed_train_data.csv

# preprocessed_batch_data:
#   type: pandas.CSVDataset
#   filepath: data/03_primary/preprocessed_batch_data.csv

# trained_standard_scaler:
#   type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
#   dataset:
#     type: pickle.PickleDataset
#     filepath: data/04_feature/scaler.pkl

# trained_one_hot_encoder:
#   type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
#   dataset:
#     type: pickle.PickleDataset
#     filepath: data/04_feature/encoder.pkl
