import pytest
import pandas as pd
import numpy as np
import mlflow
from unittest.mock import patch, MagicMock
from src.mlops_project.pipelines._08_model_selection.nodes import model_selection
from src.mlops_project.pipelines._08_model_selection.pipeline import create_pipeline 


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5],
        "feat2": [5, 4, 3, 2, 1],
    })
    X_test = pd.DataFrame({
        "feat1": [6, 7],
        "feat2": [0, -1],
    })
    y_train = pd.Series([0, 1, 0, 1, 0])
    y_test = pd.Series([1, 0])
    return X_train, X_test, y_train, y_test

@pytest.fixture
def parameters():
    return {
        'hyperparameters': {
            'LogisticRegression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 10, 'log': True},
                'max_iter': {'type': 'int', 'low': 50, 'high': 200}
            },
            'GaussianNB': {},
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 20},
                'max_depth': {'type': 'int', 'low': 1, 'high': 5},
            },
            'GradientBoostingClassifier': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5}
            }
        }
    }



def test_get_or_create_experiment_id_creates_new_experiment():
    with patch('mlflow.get_experiment_by_name', return_value=None), \
         patch('mlflow.create_experiment', return_value="new_exp_id"), \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.last_active_run') as mock_last_run, \
         patch('mlflow.sklearn.autolog'):

        # Mock the context manager for start_run
        mock_run = MagicMock()
        mock_run.__enter__.return_value = mock_run
        mock_run.__exit__.return_value = None
        mock_start_run.return_value = mock_run

        # Mock the return of last_active_run
        mock_last_run_instance = MagicMock()
        mock_last_run_instance.info.run_id = "mock_run_id"
        mock_last_run.return_value = mock_last_run_instance

        # Dummy data
        X_train = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [3, 4, 5, 6]
        })
        X_test = pd.DataFrame({'a': [5], 'b': [6]})
        y_train = pd.Series([0, 1, 0, 1])  
        y_test = pd.Series([0])

        parameters = {
            'hyperparameters': {
                'LogisticRegression': {
                    'C': {'type': 'float', 'low': 0.01, 'high': 10.0, 'log': True}
                },
                'GaussianNB': {
                    'var_smoothing': {'type': 'float', 'low': 1e-11, 'high': 1e-7}
                },
                'RandomForestClassifier': {
                    'max_depth': {'type': 'int', 'low': 2, 'high': 5},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True},
                    'min_samples_split': {'type': 'float', 'low': 0.1, 'high': 0.5},
                    'criterion': {'type': 'categorical', 'values': ['gini', 'entropy']},
                    'some_param': {'type': 'other', 'values': [42]}
                },
                'GradientBoostingClassifier': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 150},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
                    'max_depth': {'type': 'int', 'low': 2, 'high': 5}
                }
            }
        }


        # Run your function
        model_selection(X_train, X_test, y_train, y_test, parameters)


def test_model_selection_basic(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    
    parameters = {
        'hyperparameters': {
            'LogisticRegression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 10, 'log': True},
                'max_iter': {'type': 'int', 'low': 50, 'high': 200}
            },
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 10},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5}
            },
            'GradientBoostingClassifier': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5}
            },
            'GaussianNB': {}
        }
    }
    
    model = model_selection(X_train, X_test, y_train, y_test, parameters)


def test_model_selection_full_flow():
    X_train = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20),
    })
    y_train = pd.Series(np.random.randint(0, 2, 20))
    X_test = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
    })
    y_test = pd.Series(np.random.randint(0, 2, 10))

    parameters = {
        'hyperparameters': {
            'LogisticRegression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 10, 'log': True},
                'max_iter': {'type': 'int', 'low': 50, 'high': 200}
            },
            'GaussianNB': {},
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 10},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            },
            'GradientBoostingClassifier': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            }
        }
    }

    best_model = model_selection(X_train, X_test, y_train, y_test, parameters)
    from sklearn.base import BaseEstimator
    assert isinstance(best_model, BaseEstimator)


# Test the create_pipeline function
def test_create_pipeline():
    pl = create_pipeline()
    assert pl is not None
    assert hasattr(pl, "nodes")
    assert len(pl.nodes) == 1
    assert pl.nodes[0].name == "model_selection"