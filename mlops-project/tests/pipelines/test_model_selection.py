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
    with patch('mlflow.get_experiment_by_name', return_value=None) as mock_get_exp, \
         patch('mlflow.create_experiment', return_value="new_exp_id") as mock_create_exp, \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.sklearn.autolog') as mock_autolog:
        
        # Create dummy data
        import pandas as pd
        X_train = pd.DataFrame({'a':[1,2], 'b':[3,4]})
        X_test = pd.DataFrame({'a':[5], 'b':[6]})
        y_train = pd.Series([0,1])
        y_test = pd.Series([0])
        parameters = {
        'hyperparameters': {
            'RandomForestClassifier': {
                'max_depth': {'type': 'int', 'low': 2, 'high': 5},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True},
                'min_samples_split': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'criterion': {'type': 'categorical', 'values': ['gini', 'entropy']},
                'some_param': {'type': 'other', 'values': [42]}  # this triggers the else branch
            }
        }
    }


        # Call the model_selection function
        model_selection(X_train, X_test, y_train, y_test, parameters)

        # Assert get_experiment_by_name was called
        mock_get_exp.assert_called_once()

        # Assert create_experiment called because experiment was missing
        mock_create_exp.assert_called_once()




# @patch("src.mlops_project.pipelines._08_model_selection.nodes.mlflow")
# @patch("src.mlops_project.pipelines._08_model_selection.nodes.optuna.create_study")
# def test_model_selection_basic(optuna_mock, mlflow_mock, sample_data, parameters):
#     X_train, X_test, y_train, y_test = sample_data

#     # Mock MLflow experiment retrieval/creation
#     mlflow_mock.get_experiment_by_name.return_value = MagicMock(experiment_id="exp123")
#     mlflow_mock.create_experiment.return_value = "exp123"
#     mlflow_mock.start_run.return_value.__enter__.return_value = None
#     mlflow_mock.last_active_run.return_value.info.run_id = "run123"
#     mlflow_mock.sklearn = MagicMock()

#     # Mock Optuna study and optimization
#     study_mock = MagicMock()
#     study_mock.best_params = {'n_estimators': 10, 'max_depth': 3}
#     optuna_mock.return_value = study_mock

#     def fake_optimize(func, n_trials):
#         # simulate study.optimize calling func multiple times
#         for _ in range(n_trials):
#             func(MagicMock(suggest_float=lambda n, low, high, log=False: low,
#                            suggest_int=lambda n, low, high: low,
#                            suggest_categorical=lambda n, vals: vals[0]))

#     study_mock.optimize.side_effect = fake_optimize

#     model = model_selection(X_train, X_test, y_train, y_test, parameters)

#     # Check returned model is a sklearn estimator (RandomForestClassifier here)
#     from sklearn.ensemble import RandomForestClassifier
#     assert isinstance(model, RandomForestClassifier)

#     # Confirm MLflow logging calls
#     assert mlflow_mock.start_run.call_count > 0
#     assert mlflow_mock.sklearn.log_model.called

#     # Confirm best_params in logs
#     mlflow_mock.log_params.assert_called()

#     # Confirm accuracy metric logged
#     mlflow_mock.log_metric.assert_called()


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