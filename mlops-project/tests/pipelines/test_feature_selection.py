import os
import pickle
import tempfile
import logging
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.base import BaseEstimator, ClassifierMixin
from src.mlops_project.pipelines._07_feature_selection.nodes import (
    smart_correlation_filter,
    feature_selection,
)
from src.mlops_project.pipelines._07_feature_selection.pipeline import create_pipeline 

@pytest.fixture
def correlated_df():
    # Create a DataFrame with some correlated features
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with A
        "C": [10, 9, 8, 7, 6],           # Not correlated with A or B
        "D": [10, 9, 8, 7, 6],           # Perfectly correlated with C
        "E": ["cat", "dog", "cat", "dog", "cat"]  # Non-numeric column
    }
    return pd.DataFrame(data)

def test_smart_correlation_filter_removes_high_corr(correlated_df):
    # Select only numeric columns before passing to smart_correlation_filter
    numeric_df = correlated_df.select_dtypes(include=[np.number])

    filtered_df, dropped = smart_correlation_filter(numeric_df, high_corr_thresh=0.9, relevance_corr_thresh=0.5)
    
    # B is highly correlated with A, one should be dropped (B likely)
    assert "B" in dropped or "A" in dropped
    
    # D and C perfectly correlated, one should be dropped
    assert "D" in dropped or "C" in dropped

    # Since numeric_df was passed, 'E' is not in filtered_df at all
    assert "E" not in filtered_df.columns



def test_feature_selection_drops_high_corr_and_rfe(monkeypatch, correlated_df):
    y = pd.Series([0, 1, 0, 1, 0])

    # Patch open + pickle.load to simulate loading a classifier model
    dummy_classifier = MagicMock()
    dummy_classifier.fit.return_value = dummy_classifier
    dummy_classifier.get_params.return_value = {}
    
    class DummyRFE:
        def __init__(self, estimator): pass
        def fit(self, X, y):
            self.support_ = np.array([True] * X.shape[1])  # or any mask with length == X.shape[1]
            return self
        def get_support(self, arg):
            return self.support_

    monkeypatch.setattr("builtins.open", lambda *a, **k: tempfile.TemporaryFile())
    monkeypatch.setattr("pickle.load", lambda f: dummy_classifier)
    monkeypatch.setattr("src.mlops_project.pipelines._07_feature_selection.nodes.RFE", DummyRFE)

    params = {
        "drop_highly_correlated": True,
        "correlation_threshold": 0.9,
        "relevance_correlation_threshold": 0.5,
        "feature_selection": "rfe",
        "baseline_model_params": {},
    }
    
    selected_cols = feature_selection(correlated_df, y, params)

    # Columns returned should be subset of numeric columns (excluding highly correlated dropped)
    assert isinstance(selected_cols, list)
    assert all(isinstance(col, str) for col in selected_cols)
    assert len(selected_cols) > 0

def test_feature_selection_no_method(monkeypatch, correlated_df):
    y = pd.Series([0, 1, 0, 1, 0])

    params = {
        "drop_highly_correlated": False,
        "feature_selection": "none",
    }

    # Should return all numeric columns since no matching feature_selection method
    selected_cols = feature_selection(correlated_df, y, params)

    expected_cols = correlated_df.select_dtypes(include=[np.number]).columns.tolist()
    assert set(selected_cols) == set(expected_cols)



def test_feature_selection_load_model_failure(monkeypatch, correlated_df):
    y = pd.Series([0, 1, 0, 1, 0])

    def open_fail(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("builtins.open", open_fail)

    class DummyClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.feature_importances_ = np.ones(X.shape[1])
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])

    monkeypatch.setattr(
        "src.mlops_project.pipelines._07_feature_selection.nodes.RandomForestClassifier",
        lambda **kwargs: DummyClassifier()
    )
    
    params = {
        "drop_highly_correlated": False,
        "feature_selection": "rfe",
        "baseline_model_params": {"n_estimators": 10},
    }

    selected_cols = feature_selection(correlated_df, y, params)

    assert isinstance(selected_cols, list)
    assert len(selected_cols) > 0



# Test the create_pipeline function
def test_create_pipeline():
    pl = create_pipeline()
    assert pl is not None
    assert hasattr(pl, "nodes")
    assert len(pl.nodes) == 1
    assert pl.nodes[0].name == "model_feature_selection"