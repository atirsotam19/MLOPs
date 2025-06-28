import pytest
import pandas as pd
import numpy as np

from mlops_project.pipelines._05_2_data_unit_tests_feature_engineering.nodes import (
    validate_engineered_features
)


@pytest.fixture
def valid_feature_engineered_df():
    return pd.DataFrame({
        "total_assets": [1000000, 2000000],
        "loan_to_assets_ratio": [0.1, 0.3],
        "debt_to_income_ratio": [0.5, 1.2],
        "loan_score": [350.0, 480.0],
        "assets_per_dependent": [500000, 1000000],
        "loan_amount_per_month": [10000, 15000],
        "assets_minus_loan": [800000, 1700000],
        "is_high_risk_profile": [0, 1],
        "loan_term_group": ["medium", "short"],
        "cibil_score_bin": ["medium", "low"],
        "is_graduate_and_employed": [1, 0]
    })


def test_valid_data_passes(valid_feature_engineered_df):
    """Should pass with correct data."""
    result = validate_engineered_features(valid_feature_engineered_df)
    assert isinstance(result, pd.DataFrame)
    assert not result.isnull().any().any()


def test_missing_columns_raises(valid_feature_engineered_df):
    """Should raise ValueError if required columns are missing."""
    df_missing = valid_feature_engineered_df.drop(columns=["loan_score"])
    with pytest.raises(ValueError, match="Missing engineered columns"):
        validate_engineered_features(df_missing)


def test_nan_raises(valid_feature_engineered_df):
    """Should raise ValueError if NaNs are present."""
    df_nan = valid_feature_engineered_df.copy()
    df_nan.loc[0, "total_assets"] = np.nan
    with pytest.raises(ValueError, match="NaNs found in engineered features"):
        validate_engineered_features(df_nan)


def test_inf_raises(valid_feature_engineered_df):
    """Should raise ValueError if infinite values are present."""
    df_inf = valid_feature_engineered_df.copy()
    df_inf.loc[1, "loan_to_assets_ratio"] = np.inf
    with pytest.raises(ValueError, match="Infinite values found"):
        validate_engineered_features(df_inf)


def test_out_of_bounds_ratio_raises(valid_feature_engineered_df):
    """Should assert if ratio is too high."""
    df_high_ratio = valid_feature_engineered_df.copy()
    df_high_ratio["loan_to_assets_ratio"] = [0.1, 999]
    with pytest.raises(AssertionError, match="loan_to_assets_ratio out of bounds"):
        validate_engineered_features(df_high_ratio)


def test_invalid_bin_labels_raises(valid_feature_engineered_df):
    """Should raise ValueError if invalid bin labels are used."""
    df_invalid_bin = valid_feature_engineered_df.copy()
    df_invalid_bin["cibil_score_bin"] = ["banana", "low"]
    with pytest.raises(ValueError, match="Invalid cibil_score_bin values"):
        validate_engineered_features(df_invalid_bin)
