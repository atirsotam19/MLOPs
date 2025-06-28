import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.mlops_project.pipelines._09_model_train.nodes import model_train 

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        "feat1": [0, 1, 0, 1],
        "feat2": [1, 0, 1, 0],
        "feat3": [0.5, 0.7, 0.1, 0.3]
    })
    X_test = pd.DataFrame({
        "feat1": [1, 0],
        "feat2": [0, 1],
        "feat3": [0.6, 0.2]
    })
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([1, 0])
    parameters = {
        "baseline_model_params": {"n_estimators": 5, "random_state": 1},
        "use_feature_selection": False
    }
    best_columns = ["feat1", "feat2", "feat3"]
    return X_train, X_test, y_train, y_test, parameters, best_columns

@patch("mlflow.get_experiment_by_name")
@patch("mlflow.start_run")
@patch("mlflow.log_metric")
@patch("mlflow.last_active_run")
def test_model_train_basic(
    mock_last_active_run, 
    mock_log_metric, 
    mock_start_run, 
    mock_get_experiment_by_name, 
    sample_data
):
    # Mock experiment returned by mlflow.get_experiment_by_name
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "12345"
    mock_get_experiment_by_name.return_value = mock_experiment

    # Mock mlflow run context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "fake_run_id"
    mock_start_run.return_value.__enter__.return_value = mock_run
    mock_last_active_run.return_value = mock_run

    X_train, X_test, y_train, y_test, parameters, best_columns = sample_data

    model, columns, results_dict, plt_obj, results_dict_2 = model_train(
        X_train, X_test, y_train, y_test, parameters, best_columns
    )

    # verify outputs
    assert "train_score" in results_dict
    assert "test_score" in results_dict
    assert len(columns) == len(best_columns)

