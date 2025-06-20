import pytest
import pandas as pd
from src.mlops_project.pipelines._01_data_unit_tests.nodes import validate_data

@pytest.fixture
def valid_dataframe():
    """Provides a mock dataframe with valid data to pass expectations."""
    return pd.DataFrame({
        "graduate": [1, 0, 1],
        "self_employed": [1, 0, 1],
        "loan_approved": [1, 0, 1],
        "no_of_dependents": [1, 2, 3],
        "loan_term": [10, 12, 14],
        "income_annum": [500000, 1000000, 1500000],
        "loan_amount": [1500000, 1600000, 1700000],
        "residential_assets_value": [200000, 300000, 250000],
        "commercial_assets_value": [50000, 100000, 150000],
        "luxury_assets_value": [1400000, 1500000, 1600000],
        "bank_asset_value": [200500, 300000, 400000],
        "cibil_score": [700, 750, 800],
    })


def test_data_expectations(valid_dataframe):
    """Test the `test_data` node runs without assertion errors."""
    validation_df = validate_data(valid_dataframe)

    # Check if returned dataframe is not empty
    assert not validation_df.empty, "Validation results should not be empty."

    # Check if all expectations passed
    assert validation_df["Success"].all(), "Not all expectations passed."