import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

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
    })
    y_train = pd.Series([0, 1, 0, 1, 0])
    y_test = pd.Series([1, 0])
    best_cols = ["feat1", "feat2", "feat3"]
    return X_train, X_test, y_train, y_test, best_cols

@patch("builtins.open")
@patch("pickle.load", side_effect=Exception("no model"))
@patch("mlflow.start_run")
@patch("mlflow.get_experiment_by_name")
@patch("yaml.load", return_value={"tracking": {"experiment": {"name": "test"}}})
@patch("shap.TreeExplainer")
@patch("shap.summary_plot")
def test_model_train_runs(
    mock_shap_plot,
    mock_shap_explainer,
    mock_yaml,
    mock_get_exp,
    mock_mlflow_run,
    mock_pickle_load,
    mock_open,
    sample_data,
):
    X_train, X_test, y_train, y_test, best_cols = sample_data

    mock_model = MagicMock()
    mock_model.fit.return_value = mock_model
    mock_model.predict.side_effect = lambda X: np.random.randint(0, 2, size=len(X))
    mock_shap_explainer.return_value.return_value = np.random.randn(len(X_train), len(X_train.columns))

    mock_get_exp.return_value = MagicMock(experiment_id="1234")

    params = {
        "baseline_model_params": {"n_estimators": 10},
        "use_feature_selection": True
    }

    model, cols, metrics, plot, full_metrics = model_train(
        X_train, X_test, y_train, y_test, params, best_cols
    )

    assert model is not None
    assert isinstance(cols, pd.Index)
    assert isinstance(metrics, dict)
    assert "train_score" in metrics
