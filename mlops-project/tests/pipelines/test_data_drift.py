import os
import pandas as pd
import numpy as np
import tempfile

from kedro.pipeline import Pipeline
from src.mlops_project.pipelines._12_data_drift.nodes import (
    compute_psi,
    plot_psi_bar,
    plot_drift_for_features,
)
from src.mlops_project.pipelines._12_data_drift.pipeline import create_pipeline


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
    assert all(psi_df["PSI"] < 0.1)


def test_compute_psi_with_drift():
    reference = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 1, 100),
    })
    current = pd.DataFrame({
        "feature1": np.random.normal(5, 1, 100),
        "feature2": np.random.normal(10, 1, 100),
    })

    features = ["feature1", "feature2"]
    psi_df = compute_psi(reference, current, features, bins=10)

    assert isinstance(psi_df, pd.DataFrame)
    assert all(psi_df.index == features)
    assert all(psi_df["PSI"] >= 0.1)


def test_plot_psi_bar_creates_file():
    psi_df = pd.DataFrame({
        "PSI": [0.02, 0.15, 0.4]
    }, index=["feature1", "feature2", "feature3"])

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_psi_bar(psi_df, tmpdir)

        output_path = os.path.join(tmpdir, "psi_bar_plot.png")
        assert os.path.exists(output_path)
        assert os.path.isfile(output_path)


def test_plot_drift_for_features_creates_files():
    # Generate synthetic reference and current data
    reference = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 1, 100),
    })

    current = pd.DataFrame({
        "feature1": np.random.normal(1, 1, 300),
        "feature2": np.random.normal(6, 1, 300),
    })

    features = ["feature1", "feature2"]
    chunk_size = 100
    threshold = 0.2

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_drift_for_features(current, reference, features, chunk_size, tmpdir, threshold)

        # Check that drift plots were created
        for feature in features:
            output_path = os.path.join(tmpdir, f"{feature}_drift.png")
            assert os.path.exists(output_path)
            assert os.path.isfile(output_path)


def test_create_pipeline_returns_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 3  # compute_psi, plot_psi_bar, plot_drift_for_features
