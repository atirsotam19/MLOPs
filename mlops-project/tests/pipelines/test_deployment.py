import os
import json
import pickle
import joblib
import tempfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlops_project.pipelines._11_deployment.nodes import (
    save_production_model,
    save_production_columns,
    save_production_metrics,
)

# Test cases for the deployment nodes in the MLOps project
def test_save_production_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit([[0, 1], [1, 0]], [0, 1])

    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name

    # Call the function to save the model
    returned_model = save_production_model(model, path)

    # Check if the file was created
    assert os.path.exists(path)

    # Check if the saved model can be loaded and used
    loaded_model = joblib.load(path)
    assert hasattr(loaded_model, "predict")
    
    # Confirm the returned object is the same as the input
    assert returned_model is model

    os.remove(path)

# Test cases for saving production columns and metrics
def test_save_production_columns():
    columns = pd.Index(["feature1", "feature2"])

    # Temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name

    # Call the function to save the columns
    returned_columns = save_production_columns(columns, path)

    assert os.path.exists(path)

    # Load and verify the saved columns
    with open(path, "rb") as f:
        loaded_columns = pickle.load(f)

    assert all(loaded_columns == columns)
    assert returned_columns is columns

    os.remove(path)

# Test cases for saving production metrics
def test_save_production_metrics():
    metrics = {"accuracy": 0.95, "f1_score": 0.9}

    # Temporary file path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    # Call the function to save metrics
    returned_metrics = save_production_metrics(metrics, path)

    assert os.path.exists(path)

    # Load and verify the saved metrics
    with open(path, "r") as f:
        loaded_metrics = json.load(f)

    assert loaded_metrics == metrics    # Confirm the returned object is the same as the input
    assert returned_metrics is metrics  

    os.remove(path)
