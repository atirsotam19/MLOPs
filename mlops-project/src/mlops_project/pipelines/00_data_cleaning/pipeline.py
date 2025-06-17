from kedro.pipeline import Pipeline, node
from .nodes import clean_data  # assuming your cleaning function is named clean_data in nodes.py

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_data",  # name of the dataset input in catalog
                outputs="cleaned_data",  # name of the cleaned dataset output
                name="clean_data_node",
            )
        ]
    )
