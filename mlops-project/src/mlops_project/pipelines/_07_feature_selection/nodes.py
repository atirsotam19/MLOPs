import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def smart_correlation_filter(df: pd.DataFrame, high_corr_thresh: float = 0.9, relevance_corr_thresh: float = 0.5):
    """
    Drops highly correlated variables, preferring to keep the one with fewer strong correlations.
    """
    corr_matrix = df.corr().abs()
    to_drop = set()
    considered = set()

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 == col2 or (col2, col1) in considered:
                continue

            considered.add((col1, col2))
            if corr_matrix.loc[col1, col2] > high_corr_thresh:
                # Sum of moderate correlations with other features
                col1_corr_sum = corr_matrix[col1][(corr_matrix[col1] > relevance_corr_thresh) & (corr_matrix.index != col2)].sum()
                col2_corr_sum = corr_matrix[col2][(corr_matrix[col2] > relevance_corr_thresh) & (corr_matrix.index != col1)].sum()

                if col1_corr_sum >= col2_corr_sum:
                    to_drop.add(col1)
                else:
                    to_drop.add(col2)

    return df.drop(columns=list(to_drop)), list(to_drop)


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]):
    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    # Keep only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numeric = X_train[numeric_cols]
    log.info(f"After dropping non-numeric cols: {len(X_train_numeric.columns)} columns")

    dropped_cols = []
    if parameters.get("drop_highly_correlated", False):
        threshold = parameters.get("correlation_threshold", 0.9)
        relevance_threshold = parameters.get("relevance_correlation_threshold", 0.5)
        X_train_numeric, dropped_cols = smart_correlation_filter(X_train_numeric, threshold, relevance_threshold)
        log.info(f"Smart correlation filter dropped {len(dropped_cols)} columns: {dropped_cols}")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)

        # Try to load an existing model, else create a new one
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
            log.info("Loaded existing model for RFE.")
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])
            log.warning("Could not load model, using RandomForestClassifier from scratch.")

        rfe = RFE(classifier)
        rfe = rfe.fit(X_train_numeric, y_train)
        support_mask = rfe.get_support(1)
        X_cols = X_train_numeric.columns[support_mask].tolist()
    else:
        log.warning("No feature selection method matched. Returning all columns.")
        X_cols = list(X_train_numeric.columns)

    log.info(f"Number of best columns is: {len(X_cols)}")

    return X_cols