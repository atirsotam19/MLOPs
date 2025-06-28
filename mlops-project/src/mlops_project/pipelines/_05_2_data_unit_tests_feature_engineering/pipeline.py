
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import validate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= validate_data,
                inputs="cleaned_loans",
                outputs= "reporting_tests",
                name="data_unit_tests",
            ),

        ]
    )
