import pandas as pd
import pytest
import sys
import os
from src.mlops_project.pipelines._00_data_cleaning.nodes import clean_data

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        " education": [" Graduate ", "Not Graduate"],
        "self_employed": [" Yes", "No "],
        "loan_status": ["Approved ", " Rejected"],
        "no_of_dependents": [1, 2],
        "loan_term": [360, 120]
    })

def test_clean_data_output_structure(sample_raw_data):
    cleaned_df = clean_data(sample_raw_data.copy())

    # Check if datetime column is present
    assert "datetime" in cleaned_df.columns

    # Check dropped columns
    assert "education" not in cleaned_df.columns
    assert "loan_status" not in cleaned_df.columns

    # Check mapped columns
    assert "graduate" in cleaned_df.columns
    assert cleaned_df["graduate"].tolist() == [1, 0]

    assert "self_employed" in cleaned_df.columns
    assert cleaned_df["self_employed"].tolist() == [1, 0]

    assert "loan_approved" in cleaned_df.columns
    assert cleaned_df["loan_approved"].tolist() == [1, 0]

    # Check type conversion to string
    assert cleaned_df["no_of_dependents"].dtype == object
    assert cleaned_df["loan_term"].dtype == object

    assert all([col == col.strip() for col in cleaned_df.columns])