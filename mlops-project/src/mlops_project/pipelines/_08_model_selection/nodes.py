"""
This is a boilerplate pipeline '_08_model_selection'
generated using Kedro 0.19.14
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import optuna
import warnings
warnings.filterwarnings("ignore", category=Warning)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

import mlflow

logger = logging.getLogger(__name__)

def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id
     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):
    
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        sklearn.base.BaseEstimator:
            The selected model (new champion if better than current one, otherwise the existing champion model).
    """
   
    models_dict = {
        'LogisticRegression': LogisticRegression(),
        'GaussianNB': GaussianNB(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }

    initial_results = {}   

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)
        logger.info(experiment_id)


    logger.info('Starting first step of model selection : Comparing between model types')

    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):
            #mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train = np.ravel(y_train)
            model.fit(X_train, y_train)
            initial_results[model_name] = model.score(X_test, y_test)
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")
    
    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    ##### HYPERPARAMETER TUNING WITH GRIDSEARCH #####

    # logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    # logger.info('Starting second step of model selection : Hyperparameter tuning')

    # # Perform hyperparameter tuning with GridSearchCV
    # param_grid = parameters['hyperparameters'][best_model_name]
    # with mlflow.start_run(experiment_id=experiment_id,nested=True):
    #     gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    #     gridsearch.fit(X_train, y_train)
    #     best_model = gridsearch.best_estimator_


    # logger.info(f"Hypertunned model score: {gridsearch.best_score_}")
    # pred_score = accuracy_score(y_test, best_model.predict(X_test))

    # if champion_dict['test_score'] < pred_score:
    #     logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
    #     return best_model
    # else:
    #     logger.info(f"Champion model is still {champion_dict['regressor']} with score: {champion_dict['test_score']} vs {pred_score} ")
    #     return champion_model
    
    #############

    ##### HYPERPARAMETER TUNING USING OPTUNA #####

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning with Optuna')

    param_space = parameters['hyperparameters'][best_model_name]

    def objective(trial):
        params = {}
        for param_name, param_info in param_space.items():
            if param_info['type'] == 'float':
                if param_info.get('log', False):
                    params[param_name] = trial.suggest_float(param_name, param_info['low'], param_info['high'], log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, param_info['low'], param_info['high'])
            elif param_info['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_info['low'], param_info['high'])
            elif param_info['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_info['values'])
            else:
                params[param_name] = param_info['values'][0]

        model_class = type(best_model)
        model = model_class(**params)
        
        y_train_ravel = np.ravel(y_train)

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.log_params(params)
            scores = cross_val_score(model, X_train, y_train_ravel, cv=2, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            mlflow.log_metric("mean_cv_accuracy", mean_score)
        
        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    logger.info(f"Best hyperparameters found by Optuna: {best_params}")

    # Train final model with best hyperparameters on full training data
    best_model = type(best_model)(**best_params)
    best_model.fit(X_train, np.ravel(y_train))

    # Evaluate on test set
    pred_score = accuracy_score(y_test, best_model.predict(X_test))

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", pred_score)
        mlflow.sklearn.log_model(best_model, "model")

    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['regressor']} with score: {champion_dict['test_score']} vs {pred_score} ")
        return champion_model