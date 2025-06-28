import pytest
import pandas as pd
from mlops_project.pipelines._05_2_data_unit_tests_feature_engineering.nodes import validate_engineered_features

@pytest.fixture
def sample_engineered_dataframe():
    """Fixture to create a sample dataframe matching the expectation suite structure."""
    return pd.DataFrame({
        "loan_id": range(1, 6),
        "no_of_dependents": [0, 1, 2, 1, 0],
        "self_employed": ['yes', 'no', 'yes', 'no', 'yes'],
        "income_annum": [50000, 60000, 55000, 62000, 58000],
        "loan_amount": [10000, 15000, 12000, 13000, 14000],
        "loan_term": [12, 24, 36, 48, 60],
        "cibil_score": [750, 780, 700, 710, 720],
        "residential_assets_value": [20000, 25000, 22000, 23000, 24000],
        "commercial_assets_value": [5000, 6000, 5500, 5800, 5700],
        "luxury_assets_value": [1000, 1200, 1100, 1150, 1250],
        "bank_asset_value": [8000, 8500, 8300, 8700, 8600],
        "graduate": [True, True, False, True, False],
        "loan_approved": [True, False, True, False, True],
        "total_assets": [34000, 36500, 35000, 36000, 37000],
        "is_graduate_and_employed": [1, 1, 0, 1, 0],
        "loan_to_assets_ratio": [0.29, 0.41, 0.34, 0.36, 0.38],
        "cibil_score_bin": ["high", "high", "low", "medium", "medium"],
        "debt_to_income_ratio": [0.2, 0.25, 0.22, 0.26, 0.24],
        "loan_score": [0.85, 0.9, 0.75, 0.88, 0.8],
        "assets_per_dependent": [34000, 36500, 17500, 18000, 37000],
        "loan_amount_per_month": [833.33, 625.0, 333.33, 270.83, 233.33],
        "loan_term_group": ["short", "medium", "medium", "medium", "long"],
        "assets_minus_loan": [24000, 21500, 23000, 23000, 23000],
        "is_high_risk_profile": [0, 1, 0, 1, 0]
    })


def test_validate_engineered_features_all_pass(sample_engineered_dataframe):
    """Test that all validations pass for a correct engineered feature DataFrame."""
    validation_results = validate_engineered_features(sample_engineered_dataframe)

    # Assert DataFrame returned
    assert isinstance(validation_results, pd.DataFrame)
    assert not validation_results.empty

    # All tests should pass
    assert validation_results["Success"].all(), "Not all expectations passed"