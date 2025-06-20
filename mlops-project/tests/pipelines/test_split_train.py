
import pytest
import pandas as pd
import numpy as np
from src.mlops_project.pipelines._06_split_train_pipeline.nodes import split_data 
from src.mlops_project.pipelines._06_split_train_pipeline.pipeline import create_pipeline 

@pytest.fixture
def sample_df():
    # Create a small dummy dataset with a target column and loan_id
    df = pd.DataFrame(
        {
            "loan_id": [1, 2, 3, 4, 5, 6],
            "feature1": [10, 20, 30, 40, 50, 60],
            "feature2": [1, 0, 1, 0, 1, 0],
            "target": [0, 1, 0, 1, 0, 1],  # perfectly balanced
        }
    )
    return df


def test_split_data_happy_path(sample_df):
    params = {
        "target_column": "target",
        "test_fraction": 0.33,
        "random_state": 42,
    }
    X_train, X_test, y_train, y_test, cols = split_data(sample_df, params)
    # Shapes
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert list(cols) == ["feature1", "feature2"]
    # Target column should not exist in X
    assert "target" not in X_train.columns
    assert "loan_id" not in X_train.columns
    # Train and test split length
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


def test_split_data_raises_assertion_error():
    # Introduce a null in the data to hit the assertion branch
    df = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "feature1": [10, None],
            "feature2": [1, 0],
            "target": [0, 1],
        }
    )
    params = {
        "target_column": "target",
        "test_fraction": 0.5,
        "random_state": 42,
    }
    with pytest.raises(AssertionError):
        split_data(df, params)

# Test the create_pipeline function
def test_create_pipeline():
    pl = create_pipeline()
    assert pl is not None
    assert hasattr(pl, "nodes")
    assert len(pl.nodes) == 1
    assert pl.nodes[0].name == "split"



