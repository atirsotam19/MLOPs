import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.mlops_project.pipelines._10_model_predict.nodes import model_predict, evaluate_model


@pytest.fixture
def sample_data_with_target():
    return pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1, 2, 3],
        "loan_approved": [0, 1, 0]
    })


@pytest.fixture
def sample_data_without_target():
    return pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1, 2, 3],
    })


@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.predict.return_value = np.array([0, 1, 0])
    mock.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
    return mock


def test_model_predict_with_evaluation(sample_data_with_target, mock_model):
    columns = ["feature1", "feature2"]

    with patch("src.mlops_project.pipelines._10_model_predict.nodes.evaluate_model") as mock_eval:
        result_df, summary = model_predict(sample_data_with_target.copy(), mock_model, columns)

        assert "y_pred" in result_df.columns
        assert isinstance(summary, dict)
        mock_eval.assert_called_once()


def test_model_predict_without_target(sample_data_without_target, mock_model):
    columns = ["feature1", "feature2"]

    with patch("src.mlops_project.pipelines._10_model_predict.nodes.evaluate_model") as mock_eval:
        result_df, summary = model_predict(sample_data_without_target.copy(), mock_model, columns)

        assert "y_pred" in result_df.columns
        assert isinstance(summary, dict)
        mock_eval.assert_not_called()