from kedro.pipeline import Pipeline, node
from .nodes import compute_psi, plot_psi_bar

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
        ]
    )
