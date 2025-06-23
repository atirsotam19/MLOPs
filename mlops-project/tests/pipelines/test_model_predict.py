import pandas as pd
import numpy as np
import pytest
import os
from unittest.mock import MagicMock

from src.mlops_project.pipelines._10_model_predict.nodes import model_predict, evaluate_model

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'loan_approved': [0, 1, 0]  
    })

@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])
    # predict_proba deve retornar probabilidade da classe 1
    model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
    return model

@pytest.fixture
def dummy_columns():
    return ['feature1', 'feature2']

def test_model_predict_and_evaluate(dummy_data, dummy_model, dummy_columns, tmp_path):
    df_with_pred, describe = model_predict(dummy_data.copy(), dummy_model, dummy_columns, save_path=str(tmp_path))

    # Verifica se a coluna y_pred foi adicionada
    assert 'y_pred' in df_with_pred.columns
    assert list(df_with_pred['y_pred']) == [0, 1, 0]

    # Verifica o dicionário de descrição
    assert isinstance(describe, dict)
    assert 'feature1' in describe
    assert 'feature2' in describe

    # Testa a função evaluate_model (grava imagens no tmp_path)
    y_true = dummy_data["loan_approved"].values
    y_pred = df_with_pred["y_pred"].values
    y_proba = dummy_model.predict_proba(dummy_data[dummy_columns])[:, 1]

    evaluate_model(y_true, y_pred, y_proba, save_path=tmp_path)

    # Verifica se as imagens foram criadas
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "roc_curve.png").exists()
