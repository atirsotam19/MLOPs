import pandas as pd
import numpy as np
import pytest

from mlops_project.pipelines._04_preprocessing_train.nodes import (
    treat_outliers,
    feature_engineer,
    scale_encode,
)

@pytest.fixture
def raw_data():
    return pd.DataFrame({
        "residential_assets_value": [100000, 200000, 300000, 9999999],
        "commercial_assets_value": [150000, 250000, 350000, 8888888],
        "bank_asset_value": [50000, 60000, 70000, 7777777],
        "luxury_assets_value": [10000, 20000, 30000, 40000],
        "loan_amount": [100000, 200000, 300000, 400000],
        "income_annum": [500000, 600000, 700000, 800000],
        "graduate": [1, 1, 0, 0],
        "self_employed": [0, 1, 0, 1],
        "cibil_score": [450, 650, 750, 800],
        "no_of_dependents": [1, 2, 0, 3],
        "loan_term": [12, 18, 24, 30],
        "datetime": pd.date_range("2020-01-01", periods=4)
    })


def test_treat_outliers_clipping(raw_data):
    treated = treat_outliers(raw_data)
    assert all(treated["residential_assets_value"] <= treated["residential_assets_value"].quantile(0.75) + 1.5 * treated["residential_assets_value"].quantile(0.75) - treated["residential_assets_value"].quantile(0.25))
    assert isinstance(treated, pd.DataFrame)


def test_feature_engineer_output(raw_data):
    treated = treat_outliers(raw_data)
    engineered, summary = feature_engineer(treated)

    # Should remove 'datetime' and add engineered features
    assert "datetime" not in engineered.columns
    expected_cols = [
        "total_assets", "is_graduate_and_employed", "loan_to_assets_ratio",
        "cibil_score_bin", "debt_to_income_ratio", "loan_score", 
        "assets_per_dependent", "loan_amount_per_month", "loan_term_group", 
        "assets_minus_loan", "is_high_risk_profile"
    ]
    for col in expected_cols:
        assert col in engineered.columns

    assert isinstance(summary, dict)
    assert "loan_amount" in summary  # summary stats


def test_scale_encode_shapes_and_outputs(raw_data):
    treated = treat_outliers(raw_data)
    engineered, _ = feature_engineer(treated)
    encoded, scaler, encoder = scale_encode(engineered)

    # Output is a DataFrame with numerical + one-hot columns
    assert isinstance(encoded, pd.DataFrame)
    assert encoded.isnull().sum().sum() == 0
    assert scaler is not None
    assert encoder is not None

    # Check that one-hot encoded columns exist
    one_hot_columns = [col for col in encoded.columns if "cibil_score_bin" in col or "loan_term_group" in col]
    assert len(one_hot_columns) > 0
