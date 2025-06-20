"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)

def build_expectation_suite(expectation_suite_name: str, feature_group: str, features: list = None) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
             
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
    
    expectation_suite_bank = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    

    # numerical features
    if feature_group == 'numerical_features':
        # Numerical columns with float type
        float_cols = [
            'income_annum',
            'loan_amount',
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value',
        ]

        for col in float_cols:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"},
                )
            )
        
            # Minumum value accepted
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": col,
                        "min_value": 0,
                        "strict_min": False,
                    },
                )
            )

        # Numerical columns with int type
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "cibil_score", "type_": "int64"},
            )
        )

    # Categorical features

    if feature_group == 'categorical_features':
        
        # Categorical columns with binary values
        
        binary_columns = ['graduate', 'self_employed', 'loan_approved']
        for col in binary_columns:
            if features and col not in features:
                continue
            # Checks if the values are 0 or 1
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": col, "value_set": [0, 1]},
                )
            )
            # Checks if the values are not null
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col},
                )
            )
        # Number of dependents
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={
                    "column": "no_of_dependents",
                    "value_set": list(range(0, 11)),  # 0 a 10
                },
            )
        )
        # Loan Term
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "loan_term",
                    "min_value": 2.0,
                    "max_value": 20.0,
                },
            )
        )

    # Target Expectations
    if feature_group == 'target':
    
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={
                    "column": "loan_approved",
                    "value_set": [0, 1],
                },
            )
        )
        # The target values should not be null
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "loan_approved"},
            )
        )

    return expectation_suite_bank

import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    
    
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["loan_id"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.

    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def ingestion(
    df: pd.DataFrame,
    parameters: Dict[str, Any]):

    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
       
    
    
    """
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.drop_duplicates()

    logger.info(f"The dataset contains {len(df.columns)} columns.")

    numerical_features = ["bank_asset_value", "commercial_assets_value", "residential_assets_value", "cibil_score", "loan_amount", "income_annum", "luxury_assets_value"]
    categorical_features = ["self_employed", "no_of_dependents", "loan_term", "graduate", "loan_approved"]

    if parameters["target_column"] in categorical_features:
        categorical_features.remove(parameters["target_column"])

    

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations","categorical_features", features=categorical_features)
    validation_expectation_suite_target = build_expectation_suite("target_expectations","target")

    numerical_feature_descriptions = [
    {"name": "loan_id",
     "description": "Unique identifier for the loan application."},
    {"name": "datetime",
     "description": "Datetime stamp of the application."},
    {"name": "bank_asset_value",
     "description": "Total value of the applicant's assets held in bank accounts, in monetary units."},
    {"name": "cibil_score",
     "description": "Credit score of the applicant, ranging from 300 to 900."},
    {"name": "commercial_assets_value",
     "description": "Total value of the applicant's commercial property assets, in monetary units."},
    {"name": "income_annum",
     "description": "Annual income of the applicant, in monetary units."},
    {"name": "loan_amount",
     "description": "Total loan amount requested by the applicant, in monetary units."},
    {"name": "luxury_assets_value",
     "description": "Estimated value of luxury assets owned by the applicant."},
    {"name": "residential_assets_value",
     "description": "Total value of residential properties owned by the applicant, in monetary units."}
    ]

    categorical_feature_descriptions = [
    {"name": "loan_id",
     "description": "Unique identifier for the loan application."},
    {"name": "datetime",
     "description": "Datetime stamp of the application."},
    {"name": "graduate",
     "description": "Indicates whether the applicant is a graduate (1) or not (0)."},
    {"name": "loan_term",
     "description": "Duration of the loan term in years, ranging typically from 2 to 20."},
    {"name": "no_of_dependents",
     "description": "Number of dependents the applicant has."},
    {"name": "self_employed",
     "description": "Indicates whether the applicant is self-employed (1) or not (0)."}
    ]

    target_feature_descriptions = [
    {"name": "loan_id",
     "description": "Unique identifier for the loan application."},
    {"name": "datetime",
     "description": "Datetime stamp of the application."},
    {"name": "loan_approved",
     "description": "Target variable indicating whether the loan was approved (1) or not (0)."}
    ]



    df_numeric = df[["loan_id","datetime"] + numerical_features]
    df_categorical = df[["loan_id","datetime"] + categorical_features]
    df_target = df[["loan_id","datetime"] + [parameters["target_column"]]]


    if parameters["to_feature_store"]:

        object_fs_numerical_features = to_feature_store(
            df_numeric,"numerical_features",
            1,"Numerical Features",
            numerical_feature_descriptions,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )

        object_fs_categorical_features = to_feature_store(
            df_categorical,"categorical_features",
            1,"Categorical Features",
            categorical_feature_descriptions,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )

        object_fs_target_features = to_feature_store(
            df_target,"target_features",
            1,"Target Features",
            target_feature_descriptions,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )

    return df