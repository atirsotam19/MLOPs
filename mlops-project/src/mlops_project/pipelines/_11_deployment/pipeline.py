from kedro.pipeline import Pipeline, node
from .nodes import save_production_model, save_production_columns, save_production_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=save_production_model,
                inputs=["trained_model", "params:production_model_path"],
                outputs="production_model",
                name="save_production_model_node"
            ),
            node(
                func=save_production_columns,
                inputs=["selected_features", "params:production_columns_path"],
                outputs="production_columns",
                name="save_production_columns_node"
            ),
            node(
                func=save_production_metrics,
                inputs=["model_metrics", "params:production_metrics_path"],
                outputs="production_model_metrics",
                name="save_production_metrics_node"
            ),
        ]
    )
