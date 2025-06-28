import pandas as pd
import pytest
import great_expectations as ge
from preprocessing_batch_05.nodes import feature_engineer

@pytest.fixture
def sample_input_df():
    return pd.DataFrame({
        "residential_assets_value": [100000, 500000],
        "commercial_assets_value": [200000, 300000],
        "luxury_assets_value": [50000, 100000],
        "bank_asset_value": [150000, 250000],
        "loan_amount": [100000, 200000],
        "income_annum": [500000, 1000000],
        "no_of_dependents": [1, 2],
        "loan_term": [10, 15],
        "cibil_score": [600, 750],
        "graduate": [1, 1],
        "self_employed": [0, 1],
        "datetime": ["2021-01-01", "2021-01-02"]
    })


def test_feature_engineering(sample_input_df):
    # Run the feature engineering function
    engineered_df = feature_engineer(sample_input_df)
    df = ge.from_pandas(engineered_df)

    # --- 1. Required columns ---
    expected_columns = [
        "total_assets", "loan_to_assets_ratio", "debt_to_income_ratio", "loan_score",
        "assets_per_dependent", "loan_amount_per_month", "assets_minus_loan",
        "is_high_risk_profile", "loan_term_group", "cibil_score_bin", "is_graduate_and_employed"
    ]
    for col in expected_columns:
        result = df.expect_column_to_exist(col)
        assert result.success, f"Missing expected column: {col}"

    # --- 2. Value checks ---
    assert df.expect_column_values_to_be_between("total_assets", min_value=0).success
    assert df.expect_column_values_to_be_between("loan_to_assets_ratio", min_value=0, max_value=10).success
    assert df.expect_column_values_to_be_between("debt_to_income_ratio", min_value=0, max_value=10).success
    assert df.expect_column_values_to_be_in_set("is_high_risk_profile", [0, 1]).success
    assert df.expect_column_values_to_be_in_set("is_graduate_and_employed", [0, 1]).success
    assert df.expect_column_values_to_be_in_set("cibil_score_bin", ["low", "medium", "high"]).success
    assert df.expect_column_values_to_be_in_set("loan_term_group", ["short", "medium", "long"]).success

    # --- 3. Null check ---
    for col in expected_columns:
        assert df.expect_column_values_to_not_be_null(col).success, f"Nulls found in {col}"
