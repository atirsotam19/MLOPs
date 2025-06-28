import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Run data unit tests on engineered features."""
    df = data.copy()

    # --- 1. Check required columns exist ---
    required_columns = [
        "total_assets", "loan_to_assets_ratio", "debt_to_income_ratio", "loan_score",
        "assets_per_dependent", "loan_amount_per_month", "assets_minus_loan",
        "is_high_risk_profile", "loan_term_group", "cibil_score_bin", "is_graduate_and_employed"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing engineered columns: {missing_cols}")

    # --- 2. Check for NaNs and infinite values ---
    if df[required_columns].isnull().any().any():
        raise ValueError("NaNs found in engineered features.")
    if np.isinf(df[required_columns].values).any():
        raise ValueError("Infinite values found in engineered features.")

    # --- 3. Validate value ranges (example logic) ---
    assert (df["total_assets"] >= 0).all(), "total_assets contains negative values."
    assert df["loan_to_assets_ratio"].between(0, 10).all(), "loan_to_assets_ratio out of bounds."
    assert df["debt_to_income_ratio"].between(0, 10).all(), "debt_to_income_ratio out of bounds."
    assert df["is_high_risk_profile"].isin([0, 1]).all(), "is_high_risk_profile should be 0 or 1."
    assert df["is_graduate_and_employed"].isin([0, 1]).all(), "is_graduate_and_employed should be 0 or 1."

    # --- 4. Validate binning ---
    valid_bins = ["low", "medium", "high"]
    if not df["cibil_score_bin"].isin(valid_bins).all():
        raise ValueError("Invalid cibil_score_bin values.")

    valid_terms = ["short", "medium", "long"]
    if not df["loan_term_group"].isin(valid_terms).all():
        raise ValueError("Invalid loan_term_group values.")

    logger.info("Feature engineering unit tests passed.")
    return df
