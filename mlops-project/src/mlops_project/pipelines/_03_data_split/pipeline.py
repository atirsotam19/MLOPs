
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_random


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_random,
                inputs= "cleaned_loans",           # mudar para ingested_data quando a pipeline da data ingestion funcionar
                outputs=["ref_data","ana_data"],
                name="split_out_of_sample",
            ),
        ]
    )
