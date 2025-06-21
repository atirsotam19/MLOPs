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
import os

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

    # Predict
    
    y_pred = model.predict(X[columns])
    X["y_pred"] = y_pred

    # Se tiveres y_true e y_proba no DataFrame, corre a avaliação
    if "y_true" in X.columns:
        try:
            y_proba = model.predict_proba(X[columns])[:, 1]
            evaluate_model(X["y_true"].values, y_pred, y_proba, save_path=save_path)
        except AttributeError:
            logger.warning("O modelo não tem método predict_proba. ROC não gerado.")

    describe_servings = X.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, save_path="visualizations"):
    """Gera, salva e mostra confusion matrix e ROC curve.

    Args:
    --
        y_true (np.ndarray): Labels verdadeiros.
        y_pred (np.ndarray): Previsões do modelo.
        y_proba (np.ndarray): Probabilidades para a classe positiva.
        save_path (str): Pasta onde salvar as imagens.
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

    # ROC Curve
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    roc_disp.plot()
    plt.title(f"ROC Curve (AUC={auc:.2f})")
    roc_path = os.path.join(save_path, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()