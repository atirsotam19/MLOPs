import pandas as pd
import numpy as np
import pytest

from mlops_project.pipelines._03_data_split.nodes import split_random


@pytest.fixture
def sample_df():
    """Fixture to create a simple DataFrame for testing."""
    data = {
        "loan_id": range(10),
        "amount": np.random.randint(1000, 5000, size=10),
        "status": ["approved"] * 5 + ["rejected"] * 5
    }
    return pd.DataFrame(data)


def test_split_random_row_counts(sample_df):
    ref_data, ana_data = split_random(sample_df)
    
    # Check that the total number of rows matches the inputs
    assert len(ref_data) + len(ana_data) == len(sample_df)

def test_split_random_disjoint(sample_df):
    ref_data, ana_data = split_random(sample_df)
    
    # Ensure there are no overlapping indices
    assert set(ref_data.index).isdisjoint(set(ana_data.index))

def test_split_random_ratio(sample_df):
    ref_data, ana_data = split_random(sample_df)
    
    # test 80/20 split
    assert len(ref_data) == 8
    assert len(ana_data) == 2

def test_split_random_preserves_columns(sample_df):
    ref_data, ana_data = split_random(sample_df)
    
    # Ensure columns are preserved
    assert list(ref_data.columns) == list(sample_df.columns)
    assert list(ana_data.columns) == list(sample_df.columns)
