
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                parameters: Dict[str, Any], best_columns):
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.

    Returns:
    --
        model (pickle): Trained models.
        scores (json): Trained model metrics.
    """

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    # open pickle file with regressors
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        classifier = RandomForestClassifier(**parameters['baseline_model_params'])

    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        if parameters["use_feature_selection"]:
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]
        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)
        # making predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # evaluating model
        # Accuracy score
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        
        # F1 Score Macro
        f1_train = f1_score(y_train, y_train_pred, average='macro')
        f1_test = f1_score(y_test, y_test_pred, average='macro')
        
        # Precision Score
        precision_train = precision_score(y_train, y_train_pred, average='macro')
        precision_test = precision_score(y_test, y_test_pred, average='macro')
        
        # Recall Score
        
        recall_train = recall_score(y_train, y_train_pred, average='macro')
        recall_test = recall_score(y_test, y_test_pred, average='macro')
        
        # saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = acc_train
        results_dict['test_score'] = acc_test
        results_dict['f1_train'] = f1_train
        results_dict['f1_test'] = f1_test
        results_dict['precision_train'] = precision_train
        results_dict['precision_test'] = precision_test
        results_dict['recall_train'] = recall_train
        results_dict['recall_test'] = recall_test
        
        # logging in mlflow
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)
        mlflow.log_metric("precision_train", precision_train)
        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("recall_train", recall_train)
        mlflow.log_metric("recall_test", recall_test)
        
        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"Accuracy: {acc_test}, F1-Score: {f1_test}, Precision: {precision_test}, Recall: {recall_test}")



    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    

    shap.initjs()
    # calculate shap values. This is what we will plot.
    # shap_values[:,:,1] -> since it is a classification problem, I will use SHAP for explaining the outcome of class 1.
    # you can do the same for the class 0 just by using shap_values[:,:,0]
    #shap.summary_plot(shap_values[:,:,1], X_train,feature_names=X_train.columns, show=False)    ERRO???
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)


    return model, X_train.columns, results_dict, plt, results_dict