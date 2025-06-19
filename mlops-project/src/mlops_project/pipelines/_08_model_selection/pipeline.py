"""
This is a boilerplate pipeline '_08_model_selection'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train","X_test","y_train","y_test",
                        "production_model_metrics",
                        "production_model",
                        "parameters_grid"],
                outputs="champion_model",
                name="model_selection",
            ),
        ]
    )
