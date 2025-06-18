from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_loans",           # Matches the name in catalog.yml
                outputs="cleaned_loans",      # Matches the name in catalog.yml
                name="clean_data_node",
            )
        ]
    )