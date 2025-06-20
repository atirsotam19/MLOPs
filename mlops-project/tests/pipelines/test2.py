import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
from collections import UserDict

warnings.filterwarnings("ignore", category=Warning)

from src.mlops_project.pipelines._09_model_train.nodes import model_train

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        "feat1": [6, 8, 4, 3, 6],
        "feat2": [1, 4, 5, 3, 6],
        "feat3": [1, 2, 1, 4, 3],
    })
    X_test = pd.DataFrame({
        "feat1": [2, 8],
        "feat2": [1, 9],
        "feat3": [3, 2],
    })
    y_train = pd.Series([0, 1, 0, 1, 0])
    y_test = pd.Series([1, 0])
    best_cols = ["feat1", "feat2", "feat3"]
    return X_train, X_test, y_train, y_test, best_cols

class DummyExperiment:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

@patch("builtins.open")
@patch("pickle.load", side_effect=Exception("no model"))
@patch("mlflow.log_metric") 
@patch("mlflow.start_run")
@patch("src.mlops_project.pipelines._09_model_train.nodes.mlflow.get_experiment_by_name")
@patch("yaml.load", return_value={"tracking": {"experiment": {"name": "test"}}})
@patch("shap.TreeExplainer")
@patch("shap.summary_plot")
def test_model_train_runs(
    sample_data,
    mock_shap_plot,
    mock_shap_explainer,
    mock_yaml,
    mock_get_exp,
    mock_mlflow_run,
    mock_pickle_load,
    mock_open,
):

    X_train, X_test, y_train, y_test, best_cols = sample_data

    # Mock the shap explainer
    mock_shap_explainer.return_value.return_value = np.random.randn(len(X_train), len(X_train.columns))

    # Mock the experiment
    mock_get_exp.return_value = DummyExperiment("14")

    # Mock the mlflow run context
    mock_run = MagicMock()
    mock_run.__enter__.return_value = mock_run
    mock_run.__exit__.return_value = None
    mock_run.info.run_id = "test_run_id"
    mock_mlflow_run.return_value = mock_run

    # Parameters
    params = {
        "baseline_model_params": {"n_estimators": 10},
        "use_feature_selection": True
    }

    # Call the model_train
    model, cols, metrics, plot, full_metrics = model_train(
        X_train, X_test, y_train, y_test, params, best_cols
    )

    # Assertions
    assert model is not None
    assert isinstance(cols, pd.Index)
    assert isinstance(metrics, dict)
    assert "train_score" in metrics
