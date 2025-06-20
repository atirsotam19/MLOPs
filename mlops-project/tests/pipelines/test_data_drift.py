import os
import pandas as pd
import numpy as np
import tempfile

from kedro.pipeline import Pipeline
from src.mlops_project.pipelines._12_data_drift.nodes import compute_psi, plot_psi_bar
from src.mlops_project.pipelines._12_data_drift.pipeline import create_pipeline

# Test cases for the data drift detection pipeline
def test_compute_psi_no_drift():  
    data = {
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 1, 100),
    }
    reference = pd.DataFrame(data)
    current = pd.DataFrame(data.copy()) 

    features = ["feature1", "feature2"]
    psi_df = compute_psi(reference, current, features, bins=10)

    assert isinstance(psi_df, pd.DataFrame)
    assert all(psi_df.index == features)
    assert all(psi_df["PSI"] < 0.1)  # No drift expected, PSI should be low

# Test cases for the data drift detection pipeline with drift
def test_compute_psi_with_drift():
    reference = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 1, 100),
    })
    current = pd.DataFrame({
        "feature1": np.random.normal(3, 1, 100),
        "feature2": np.random.normal(8, 1, 100),
    })

    features = ["feature1", "feature2"]
    psi_df = compute_psi(reference, current, features, bins=10)

    assert isinstance(psi_df, pd.DataFrame)
    assert all(psi_df.index == features)
    assert all(psi_df["PSI"] >= 0.1)  

# Test cases for the data drift detection pipeline with empty data
def test_plot_psi_bar_creates_file():
    psi_df = pd.DataFrame({
        "PSI": [0.02, 0.15, 0.4]
    }, index=["feature1", "feature2", "feature3"])

    # Temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_psi_bar(psi_df, tmpdir)

        output_path = os.path.join(tmpdir, "psi_bar_plot.png")
        assert os.path.exists(output_path)
        assert os.path.isfile(output_path)

def test_create_pipeline_returns_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 2  # compute_psi and plot_psi_bar nodes
