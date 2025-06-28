"""
Pipeline to validate feature engineering outputs.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_engineered_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_engineered_features,
            inputs="feature_created_batch",
            outputs="validated_feature_engineered_batch",
            name="validate_feature_engineering"
        )
    ])