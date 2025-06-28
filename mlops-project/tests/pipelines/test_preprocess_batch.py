import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from mlops_project.pipelines._05_1_preprocessing_batch.nodes import (
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


@pytest.fixture
def fitted_encoders(raw_data):
    # Run through train preprocessing to get fitted encoders
    from mlops_project.pipelines._04_preprocessing_train.nodes import (
        treat_outliers as treat_outliers_train,
        feature_engineer as feature_engineer_train,
        scale_encode as scale_encode_train,
    )

    treated = treat_outliers_train(raw_data)
    engineered, _ = feature_engineer_train(treated)
    _, scaler, encoder = scale_encode_train(engineered)
    return scaler, encoder


def test_batch_treat_outliers(raw_data):
    result = treat_outliers(raw_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == raw_data.shape
    assert not result.isnull().any().any()


def test_batch_feature_engineer(raw_data):
    treated = treat_outliers(raw_data)
    engineered = feature_engineer(treated)
    expected_cols = [
        "total_assets", "is_graduate_and_employed", "loan_to_assets_ratio",
        "cibil_score_bin", "debt_to_income_ratio", "loan_score",
        "assets_per_dependent", "loan_amount_per_month", "loan_term_group",
        "assets_minus_loan", "is_high_risk_profile"
    ]
    for col in expected_cols:
        assert col in engineered.columns
    assert "datetime" not in engineered.columns
    assert isinstance(engineered, pd.DataFrame)


def test_batch_scale_encode(raw_data, fitted_encoders):
    scaler, encoder = fitted_encoders
    treated = treat_outliers(raw_data)
    engineered = feature_engineer(treated)
    transformed = scale_encode(engineered, scaler, encoder)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == engineered.shape[0]
    assert transformed.isnull().sum().sum() == 0

    # Ensure one-hot encoded cols are present
    one_hot_cols = [col for col in transformed.columns if "cibil_score_bin" in col or "loan_term_group" in col]
    assert len(one_hot_cols) > 0
