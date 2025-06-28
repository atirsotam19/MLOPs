"""
This is a boilerplate pipeline '_10_model_predict'
generated using Kedro 0.19.14
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import mlflow
import mlflow.sklearn
import yaml

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  model: pickle.Pickler, columns: pickle.Pickler, save_path="data/08_reporting") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    """Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """

    with open("conf/local/mlflow.yml") as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        y_pred = model.predict(X[columns])
        X["y_pred"] = y_pred

        mlflow.log_param("num_features", len(columns))
        mlflow.log_dict({"input_columns": list(columns)}, "input_schema.json")

        if "loan_approved" in X.columns:
            try:
                y_proba = model.predict_proba(X[columns])[:, 1]
                evaluate_model(X["loan_approved"].values, y_pred, y_proba, save_path=save_path)
            except AttributeError:
                logger.warning("Model has no predict_proba method. ROC not generated.")

        describe_servings = X.describe().to_dict()
        mlflow.log_dict(describe_servings, "serving_statistics.json")

        logger.info("Predictions logged to MLflow.")
        return X, describe_servings

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, save_path="visualizations"):
    
    """Generate, save and show confusion matrix.

    Args:
    --
        y_true (np.ndarray): True Labels.
        y_pred (np.ndarray): Model Predictions.
        y_proba (np.ndarray): Positive class probabilities.
        save_path (str): Path where to save the images.
    """

    os.makedirs(save_path, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    cm_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    logger.info(f"Accuracy: {accuracy}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}")
