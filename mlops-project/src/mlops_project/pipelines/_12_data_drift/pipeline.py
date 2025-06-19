from kedro.pipeline import Pipeline, node
from .nodes import calculate_psi_for_all_features, plot_psi_bar
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib

encoder = joblib.load("data/04_feature/encoder.pkl")

current_encoded = encoder.transform(current_loans).toarray()
reference_encoded = encoder.transform(reference_loans).toarray()


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=calculate_psi_for_all_features,
                inputs=["current_loans_encoded", "reference_loans_encoded", "params:feature_columns", "params:psi_bins"],
                outputs="psi_df",
                name="calculate_psi_all_features_node",
            ),
            node(
                func=plot_psi_bar,
                inputs=["psi_df", "params:output_dir"],
                outputs=None,
                name="plot_psi_bar_node",
            ),
        ]
    )

