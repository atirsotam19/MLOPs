"""
This is a boilerplate pipeline 'preprocessing_batch_05'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  
from .nodes import treat_outliers, feature_engineer, scale_encode


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=treat_outliers,
            inputs="ana_data", 
            outputs="outlier_treated_batch",
            name="treat_outliers_batch"
        ),
        node(
            func=feature_engineer,
            inputs="outlier_treated_batch",
            outputs="feature_created_batch",
            name="feature_engineering_batch"
        ),
        node(
            func=scale_encode,
            inputs=["feature_created_batch", "trained_standard_scaler",
                "trained_one_hot_encoder"],
            outputs= "preprocessed_batch_data",
            name="scale_and_encode_batch"
        )
    ])
