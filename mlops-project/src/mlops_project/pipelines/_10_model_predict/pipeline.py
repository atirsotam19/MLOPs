"""
This is a boilerplate pipeline '_10_model_predict'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=["preprocessed_batch_data","production_model","production_columns", "params:save_path"],
                outputs=["df_with_predict", "predict_describe"],
                name="predict",
            ),
        ]
    )