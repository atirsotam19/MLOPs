"""
Data unit tests for the feature engineering
"""

import logging
import great_expectations as ge
import pandas as pd

logger = logging.getLogger(__name__)

def validate_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Run data unit tests on engineered features using Great Expectations."""
    df = ge.from_pandas(data)

    # --- 1. Check required columns exist ---
    required_columns = [
        "total_assets", "loan_to_assets_ratio", "debt_to_income_ratio", "loan_score",
        "assets_per_dependent", "loan_amount_per_month", "assets_minus_loan",
        "is_high_risk_profile", "loan_term_group", "cibil_score_bin", "is_graduate_and_employed"
    ]
    for col in required_columns:
        df.expect_column_to_exist(col)

    # --- 2. Check for missing or infinite values ---
    for col in required_columns:
        df.expect_column_values_to_not_be_null(col)
        df.expect_column_values_to_not_match_regex(col, r'^inf$|^-inf$')

    # --- 3. Check numeric value ranges ---
    df.expect_column_values_to_be_between("total_assets", min_value=0)
    df.expect_column_values_to_be_between("loan_to_assets_ratio", min_value=0, max_value=10)
    df.expect_column_values_to_be_between("debt_to_income_ratio", min_value=0, max_value=10)
    df.expect_column_values_to_be_in_set("is_high_risk_profile", [0, 1])
    df.expect_column_values_to_be_in_set("is_graduate_and_employed", [0, 1])

    # --- 4. Check binning labels ---
    df.expect_column_values_to_be_in_set("cibil_score_bin", ["low", "medium", "high"])
    df.expect_column_values_to_be_in_set("loan_term_group", ["short", "medium", "long"])

    logger.info("All feature engineering expectations passed.")
    return df
