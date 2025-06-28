import logging
from pathlib import Path

import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')

    df_validation = pd.DataFrame({}, columns=[
        "Success", "Expectation Type", "Column", "Column Pair", "Max Value",
        "Min Value", "Element Count", "Unexpected Count", "Unexpected Percent",
        "Value Set", "Unexpected Value", "Observed Value"
    ])

    for result in results:
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')
        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        unexpected_value = []

        if isinstance(observed_value, list):
            unexpected_value = [item for item in observed_value if item not in value_set]

        df_validation = pd.concat([
            df_validation,
            pd.DataFrame.from_dict([{
                "Success": success,
                "Expectation Type": expectation_type,
                "Column": column,
                "Column Pair": (column_A, column_B),
                "Max Value": max_value,
                "Min Value": min_value,
                "Element Count": element_count,
                "Unexpected Count": unexpected_count,
                "Unexpected Percent": unexpected_percent,
                "Value Set": value_set,
                "Unexpected Value": unexpected_value,
                "Observed Value": observed_value
            }])
        ], ignore_index=True)

    return df_validation


def validate_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    project_root = Path(__file__).parents[4]
    gx_path = project_root / "gx"
    context = gx.get_context(context_root_dir=str(gx_path))

    datasource_name = "engineered_features_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Datasource created.")
    except Exception:
        logger.info("Datasource already exists.")
        datasource = context.datasources[datasource_name]

    # Create or update expectation suite
    suite = context.add_or_update_expectation_suite("engineered_features")

    # Expect no missing values
    for column in df.columns:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column}
            )
        )

    # Specific expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "total_assets", "min_value": 0}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "loan_to_assets_ratio", "min_value": 0, "max_value": 10}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "debt_to_income_ratio", "min_value": 0, "max_value": 10}
        )
    )
    for binary_col in ["is_high_risk_profile", "is_graduate_and_employed"]:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": binary_col, "value_set": [0, 1]}
            )
        )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": "cibil_score_bin", "value_set": ["low", "medium", "high"]}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": "loan_term_group", "value_set": ["short", "medium", "long"]}
        )
    )

    # Type checks
    pd_df_ge = gx.from_pandas(df)
    type_expectations = {
        "total_assets": ["float64", "int64"],
        "loan_to_assets_ratio": ["float64", "int64"],
        "debt_to_income_ratio": ["float64", "int64"],
        "loan_score": ["float64", "int64"],
        "assets_per_dependent": ["float64", "int64"],
        "loan_amount_per_month": ["float64", "int64"],
        "assets_minus_loan": ["float64", "int64"],
        "is_high_risk_profile": ["int64"],
        "is_graduate_and_employed": ["int64"],
        "loan_term_group": ["str"],
        "cibil_score_bin": ["str"],
    }

    for col, types in type_expectations.items():
        assert pd_df_ge.expect_column_values_to_be_in_type_list(col, types).success, f"{col} failed type check."

    context.add_or_update_expectation_suite(expectation_suite=suite)

    # Validate
    data_asset_name = "engineered_data"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except Exception:
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_engineered_features",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "engineered_features",
            }
        ]
    )
    checkpoint_result = checkpoint.run()
    df_validation = get_validation_results(checkpoint_result)

    logger.info("Engineered feature unit tests passed.")
    return df_validation