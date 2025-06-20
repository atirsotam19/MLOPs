import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.mlops_project.pipelines._10_model_predict.nodes import model_predict

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    return X

@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])
    return model

@pytest.fixture
def dummy_columns():
    return ['feature1', 'feature2']

def test_model_predict(dummy_data, dummy_model, dummy_columns):
    df_with_pred, describe = model_predict(dummy_data.copy(), dummy_model, dummy_columns)

    # Verifica se a coluna y_pred foi adicionada
    assert 'y_pred' in df_with_pred.columns

    # Verifica se os valores previstos foram adicionados corretamente
    assert list(df_with_pred['y_pred']) == [0, 1, 0]

    # Verifica se describe é um dicionário
    assert isinstance(describe, dict)

    # Verifica se 'feature1' e 'feature2' estão em describe
    assert 'feature1' in describe
    assert 'feature2' in describe
