"""
This is a boilerplate pipeline 'preprocessing_train'
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
    

    logger.info("Feature engineering completed.")
    return df


def scale_encode(data: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical and encode categorical features."""
    df = data.copy()

    numerical_features = df.select_dtypes(include=["number"]).columns
    categorical_features = df.select_dtypes(include=["object", "string", "category"]).columns

    # Scale numeric features
    std_scaler = StandardScaler()
    df[numerical_features] = std_scaler.fit_transform(df[numerical_features])

    # Encode
    OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    OH_cols= pd.DataFrame(OH_encoder.fit_transform(df[categorical_features]))

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
