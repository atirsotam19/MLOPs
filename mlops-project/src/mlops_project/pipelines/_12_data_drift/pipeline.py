from kedro.pipeline import Pipeline, node
from .nodes import compute_psi, plot_psi_bar, plot_drift_for_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_psi,
                inputs=["cleaned_loans", "ref_data", "params:feature_columns", "params:psi_bins"],
                outputs="psi_df",
                name="compute_psi_node",
            ),
            node(
                func=plot_psi_bar,
                inputs=["psi_df", "params:output_dir"],
                outputs=None,
                name="plot_psi_bar_node",
            ),
            node(
                func=plot_drift_for_features,
                inputs=["cleaned_loans", "ref_data", "params:feature_columns", "params:chunk_size", "params:output_dir", "params:drift_threshold"],
                outputs=None,
                name="plot_drift_for_features_node",
            ),
        ]
    )
