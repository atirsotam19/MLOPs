
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= ingestion,
                inputs=["cleaned_loans","parameters"],
                outputs= "ingested_data",
                name="ingestion",
            ),

        ]
    )
