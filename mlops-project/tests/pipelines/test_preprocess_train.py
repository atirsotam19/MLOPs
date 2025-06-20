import pytest
import pandas as pd
import numpy as np

from mlops_project.pipelines._04_preprocessing_train.nodes import (
    treat_outliers,
    feature_engineer,
    scale_encode,
)

@pytest.fixture
def sample_data():
    # Small sample dataset with all required columns
    data = {
        "residential_assets_value": [100000, 200000, 300000, 10000000],  # one outlier
        "commercial_assets_value": [150000, 250000, 350000, 20000000],   # one outlier
        "bank_asset_value": [50000, 80000, 90000, 15000000],             # one outlier
        "luxury_assets_value": [10000, 20000, 30000, 40000],
        "graduate": [1, 0, 1, 0],
        "self_employed": [0, 1, 0, 1],
        "loan_amount": [50000, 100000, 150000, 500000],
        "cibil_score": [450, 650, 720, 800],
        "income_annum": [500000, 1000000, 750000, 2000000],
        "no_of_dependents": [0, 2, 1, 3],
        "loan_term": [10, 15, 25, 30],
        "datetime": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
    }
    return pd.DataFrame(data)

def test_treat_outliers(sample_data):
    df_out = treat_outliers(sample_data)
    # Check outliers are clipped to expected range for a known column
    col = "residential_assets_value"
    Q1 = sample_data[col].quantile(0.25)
    Q3 = sample_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # No value should be outside [lower, upper]
    assert df_out[col].max() <= upper
    assert df_out[col].min() >= lower
    # Original data unchanged (except for clipping)
    assert len(df_out) == len(sample_data)

def test_feature_engineer_output(sample_data):
    df_feat, desc = feature_engineer(sample_data)
    # Check new columns exist
    expected_cols = [
        "total_assets", "is_graduate_and_employed", "loan_to_assets_ratio", 
        "cibil_score_bin", "debt_to_income_ratio", "loan_score", 
        "assets_per_dependent", "loan_amount_per_month", "loan_term_group",
        "assets_minus_loan", "is_high_risk_profile"
    ]
    for col in expected_cols:
        assert col in df_feat.columns
    # datetime column should be dropped
    assert "datetime" not in df_feat.columns
    # describe dictionary should contain basic stats
    assert isinstance(desc, dict)
    assert "total_assets" in desc

def test_scale_encode_output_shapes(sample_data):
    df_feat, _ = feature_engineer(sample_data)
    df_scaled, scaler, encoder = scale_encode(df_feat)

    categorical_features = ["no_of_dependents", "loan_term", "cibil_score_bin", "loan_term_group"]
    
    num_df = df_feat.drop(columns=categorical_features)
    expected_num_cols = num_df.shape[1]
    expected_cat_cols = sum(len(cat) for cat in encoder.categories_)
    expected_total_cols = expected_num_cols + expected_cat_cols
    
    assert df_scaled.shape[1] == expected_total_cols

def test_scale_encode_scaler_encoder_types(sample_data):
    df_feat, _ = feature_engineer(sample_data)
    df_scaled, scaler, encoder = scale_encode(df_feat)
    
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    assert isinstance(scaler, StandardScaler)
    assert isinstance(encoder, OneHotEncoder)
    # The scaled data should be a pandas DataFrame
    assert isinstance(df_scaled, pd.DataFrame)

