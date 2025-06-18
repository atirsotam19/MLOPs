"""
This is a boilerplate pipeline 'preprocessing_batch_05'
generated using Kedro 0.19.14
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#conf_path = str(Path('') / settings.CONF_SOURCE)
#conf_loader = OmegaConfigLoader(conf_source=conf_path)
#credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

def treat_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Treat outliers using IQR method for specific columns."""
    df = data.copy()
    cols = ["residential_assets_value", "commercial_assets_value", "bank_asset_value"]

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    logger.info("Outliers treated for asset value columns.")
    return df


def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing asset value columns."""
    df = data.copy()

    # Total of assets
    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    # Is the individual graduate and not self-employed? Boolean
    df["is_graduate_and_employed"] = (df["Graduate"] & (df["self_employed"] == 0)).astype(int)

    # Loan-to-assets ratio
    df["loan_to_assets_ratio"] = df["loan_amount"] / df["total_assets"].replace(0, 1)

    # CIBIL score bin (low, medium, high)
    df["cibil_score_bin"] = pd.cut(
        df["cibil_score"],
        bins=[-float("inf"), 500, 700, float("inf")],
        labels=["low", "medium", "high"]
    )

    # Debt-to-income ratio
    df["debt_to_income_ratio"] = df["loan_amount"] / df["income_annum"].replace(0, 1)

    # Loan score formula. Need to check its worth
    df["loan_score"] = (
        (df["total_assets"] / 1_000_000)
        + (0.1 * df["cibil_score"])
        - (df["loan_amount"] / (df["income_annum"] - 500_000 * df["no_of_dependents"]).replace(0, 1))
        + (5 * df["Graduate"])
        - (3 * df["self_employed"])
    )

    # Assets-per-dependant ratio (+1 for the own individual)
    df["assets_per_dependent"] = df["total_assets"] / (df["no_of_dependents"] + 1)

    # Monthly loan payment
    df["loan_amount_per_month"] = df["loan_amount"] / df["loan_term"].replace(0, 1)

    # Loan duration bin (short, medium, long)
    df["loan_term_group"] = pd.cut(
        df["loan_term"],
        bins=[-float("inf"), 12, 20, float("inf")],
        labels=["short", "medium", "long"]
    )

    # Liquid Assets
    df["assets_minus_loan"] = df["total_assets"] - df["loan_amount"]

    # Boolean is the individual high risk? high debt or low cibil score
    df["is_high_risk_profile"] = (
        (df["cibil_score"] < 500) | (df["debt_to_income_ratio"] > 2.5)
    ).astype(int)

    logger.info("Feature engineering completed.")

    return df

def scale_encode(data: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical and encode categorical features."""
    df = data.copy()

    numerical_features = df.select_dtypes(include=["number"]).columns
    categorical_features = df.select_dtypes(include=["object", "string", "category"]).columns

    # Scale numeric features
    df[numerical_features] = std_scaler.transform(df[numerical_features])

    # Encode
    OH_cols= pd.DataFrame(OH_encoder.transform(df[categorical_features]))

    # Adding column names to the encoded data set.
    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)

    # One-hot encoding removed index; put it back
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_df = df.drop(categorical_features, axis=1)

    # Add one-hot encoded columns to numerical features
    df_final = pd.concat([num_df, OH_cols], axis=1)

    logger.info("Scaling and encoding complete.")

    logger.info(f"The final dataframe has {len(df_final.columns)} columns.")

    return df_final, std_scaler, OH_encoder
