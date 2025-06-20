import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock, mock_open
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=Warning)

# Fixtures para os dados de teste
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
    return X_train, X_test, y_train, y_test

@pytest.fixture
def best_columns():
    return ["feat1", "feat2"]

@pytest.fixture
def parameters():
    return {
        "baseline_model_params": {
            "n_estimators": 10,
            "random_state": 42
        },
        "use_feature_selection": True
    }

# Mock para o experimento MLflow
class MockExperiment:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

# Mock para a execução do MLflow
class MockRun:
    class Info:
        run_id = "mock-run-id"
    
    info = Info()

# Teste principal
@patch("mlflow.sklearn.autolog")
@patch("mlflow.start_run")
@patch("mlflow.last_active_run")
@patch("mlflow.get_experiment_by_name")
@patch("yaml.load")
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
@patch("shap.TreeExplainer")
@patch("shap.summary_plot")
def test_model_train(
    mock_shap_plot,
    mock_shap_explainer,
    mock_pickle_load,
    mock_file_open,
    mock_yaml_load,
    mock_get_exp,
    mock_last_run,
    mock_start_run,
    mock_autolog,
    sample_data,
    best_columns,
    parameters
):
    # Configura os mocks
    X_train, X_test, y_train, y_test = sample_data
    
    # Mock para o modelo
    mock_model = RandomForestClassifier(**parameters['baseline_model_params'])
    mock_pickle_load.return_value = mock_model
    
    # Mock para o MLflow
    mock_yaml_load.return_value = {"tracking": {"experiment": {"name": "test"}}}
    mock_get_exp.return_value = MockExperiment("14")
    mock_last_run.return_value = MockRun()
    
    # Mock para o SHAP
    mock_shap_explainer.return_value.return_value = np.random.rand(len(X_train), len(X_train.columns))
    
    # Importa a função a ser testada
    from src.mlops_project.pipelines._09_model_train.nodes import model_train
    
    # Executa a função
    model, columns, metrics, plot, _ = model_train(
        X_train, X_test, y_train, y_test, parameters, best_columns
    )
    
    # Verificações básicas
    assert isinstance(model, RandomForestClassifier)
    assert all(col in columns for col in best_columns)  # Verifica feature selection
    assert isinstance(metrics, dict)
    assert isinstance(plot, plt.Figure)
    
    # Verifica métricas essenciais
    required_metrics = ['accuracy_train', 'accuracy_test', 'f1_train', 'f1_test',
                       'precision_train', 'precision_test', 'recall_train', 'recall_test']
    for metric in required_metrics:
        assert metric.replace('_train', '') in str(metrics)  # Verifica se o prefixo existe
    
    # Verifica se o feature selection foi aplicado
    if parameters["use_feature_selection"]:
        assert set(columns) == set(best_columns)

# Teste quando não há feature selection
def test_model_train_no_feature_selection(sample_data, best_columns, parameters):
    parameters["use_feature_selection"] = False
    
    with patch("mlflow.sklearn.autolog"), \
         patch("mlflow.start_run"), \
         patch("mlflow.last_active_run", return_value=MockRun()), \
         patch("mlflow.get_experiment_by_name", return_value=MockExperiment("14")), \
         patch("yaml.load", return_value={"tracking": {"experiment": {"name": "test"}}}), \
         patch("builtins.open", mock_open()), \
         patch("pickle.load", return_value=RandomForestClassifier()), \
         patch("shap.TreeExplainer"), \
         patch("shap.summary_plot"):
        
        from src.mlops_project.pipelines._09_model_train.nodes import model_train
        X_train, X_test, y_train, y_test = sample_data
        
        model, columns, _, _, _ = model_train(
            X_train, X_test, y_train, y_test, parameters, best_columns
        )
        
        # Verifica se todas as colunas originais estão presentes
        assert set(columns) == set(X_train.columns)

# Teste quando o champion_model.pkl não existe
def test_model_train_no_champion_model(sample_data, best_columns, parameters):
    with patch("builtins.open", side_effect=FileNotFoundError()), \
         patch("mlflow.sklearn.autolog"), \
         patch("mlflow.start_run"), \
         patch("mlflow.last_active_run", return_value=MockRun()), \
         patch("mlflow.get_experiment_by_name", return_value=MockExperiment("14")), \
         patch("yaml.load", return_value={"tracking": {"experiment": {"name": "test"}}}), \
         patch("shap.TreeExplainer"), \
         patch("shap.summary_plot"):
        
        from src.mlops_project.pipelines._09_model_train.nodes import model_train
        X_train, X_test, y_train, y_test = sample_data
        
        model, _, _, _, _ = model_train(
            X_train, X_test, y_train, y_test, parameters, best_columns
        )
        
        # Verifica se um novo modelo foi criado com os parâmetros padrão
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == parameters['baseline_model_params']['n_estimators']