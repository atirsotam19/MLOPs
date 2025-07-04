"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

import os

logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
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
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation


def validate_data(df):
    # Define project root (two levels up from nodes.py)
    project_root = Path(__file__).parents[4]

    # Construct absolute path to your gx directory
    gx_path = project_root / "gx"

    # Pass as string to Great Expectations context
    context = gx.get_context(context_root_dir=str(gx_path))

    datasource_name = "loans_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_loans = context.add_or_update_expectation_suite(expectation_suite_name="loans")

# EXPECTATIONS

# Categorical Expectations

# Education Expectation
    expectation_education = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "graduate",
        "value_set" : [0, 1]
    },
        )
    suite_loans.add_expectation(expectation_configuration=expectation_education)
    
# Self Employed Expectation
    expectation_self_employed = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "self_employed",
        "value_set" : [0, 1]
    },
        )
    suite_loans.add_expectation(expectation_configuration=expectation_self_employed)

# Loan Status Expectation
    if "loan_approved" in df.columns and df["loan_approved"].notnull().any():
        expectation_loan_approved = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "loan_approved",
            "value_set" : [0, 1]
        },
            )
        suite_loans.add_expectation(expectation_configuration=expectation_loan_approved)
        
        expectation_loan_approved_not_null = ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={
            "column": "loan_approved"
        },
            )
        suite_loans.add_expectation(expectation_configuration=expectation_loan_approved_not_null)

# Number of Dependents Expectation   
    expectation_dependents = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "no_of_dependents",
            "value_set": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_dependents)

# Loan Term Expectation
    expectation_loan_term = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "loan_term",
            "value_set": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_loan_term)

# Continuos Expectation

# Income Annum
    expectation_income_annum = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "income_annum",
            "min_value": 200000.0,
            "max_value": 10000000.0,  
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_income_annum)
    expectation_income_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "income_annum",
            "min_value": 848864,      # mean - 1.5*std
            "max_value": 9269384,     # mean + 1.5*std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_income_mean)
    
# Loan Amount

    q_low_loan_amount = 700000.0
    q_high_loan_amount = 35700000.0

    expectation_loan_amount = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "loan_amount",
            "min_value": q_low_loan_amount,          # 1% percentile
            "max_value": q_high_loan_amount,         # 99% percentile
            "mostly": 0.98
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_loan_amount)

    expectation_loan_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "loan_amount",
            "min_value": 1568405,     # mean - 1.5 * std
            "max_value": 28698495,    # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_loan_mean)
    
# Residential Assets Value

    q_low_residential = 0.0
    q_high_residential = 25400000.0

    expectation_res_assets = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "residential_assets_value",
            "min_value": q_low_residential,        # 1% percentile
            "max_value": q_high_residential,       # 99% percentile
            "mostly": 0.98  
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_res_assets)

    expectation_res_assets_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "residential_assets_value",
            "min_value": 0,          # mean - 1.5 * std 
            "max_value": 1.7228073e7,  # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_res_assets_mean)

# Commercial Assets Value

    q_low_commercial = 0.0
    q_high_commercial = 16732000.0

    expectation_comm_assets = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "commercial_assets_value",
            "min_value": q_low_commercial,   # 1% percentile
            "max_value": q_high_commercial,  # 99% percentile
            "mostly": 0.98           
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_comm_assets)

    expectation_comm_assets_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "commercial_assets_value",
            "min_value": 0,             # mean - 1.5 * std
            "max_value": 1.1556604e7,   # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_comm_assets_mean)

# Luxury Assets Value

    q_low_luxury = 700000.0
    q_high_luxury = 36032000.0

    expectation_luxury_assets = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "luxury_assets_value",
            "min_value": q_low_luxury,   # 1% percentile
            "max_value": q_high_luxury,  # 99% percentile
            "mostly": 0.98    
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_luxury_assets)

    expectation_luxury_assets_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "luxury_assets_value",
            "min_value": 1470679,     # mean - 1.5 * std
            "max_value": 28781941,    # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_luxury_assets_mean)
    
# Bank Asset Value

    q_low_bank = 200000.0
    q_high_bank = 13100000.0

    expectation_bank_asset = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "bank_asset_value",
            "min_value": q_low_bank,    # 1% percentil
            "max_value": q_high_bank,   # 99% percentil
            "mostly": 0.98
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_bank_asset)

    expectation_bank_asset_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "bank_asset_value",
            "min_value": 101415,       # mean - 1.5 * std
            "max_value": 9851969,      # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_bank_asset_mean)

# CIBIL Score
    expectation_cibil_score = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "cibil_score",
            "min_value": 300.00,     
            "max_value": 900.00,     
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_cibil_score)

    expectation_cibil_score_mean = ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "cibil_score",
            "min_value": 341.29,  # mean - 1.5 * std
            "max_value": 858.58,  # mean + 1.5 * std
        },
    )
    suite_loans.add_expectation(expectation_configuration=expectation_cibil_score_mean)

    context.add_or_update_expectation_suite(expectation_suite=suite_loans)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe= df)
    except:
        logger.info("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe= df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_loans",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "loans",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)
    #base on these results you can make an assert to stop your pipeline

    pd_df_ge = gx.from_pandas(df)

    # Categorical or binary variables (expected as int or str)
    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "no_of_dependents", ["int64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "self_employed", ["int64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "graduate", ["int64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "loan_approved", ["int64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "loan_term", ["int64"]
    ).success == True

    # Continuous/Numeric variables (allow int and float)
    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "income_annum", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "loan_amount", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "residential_assets_value", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "commercial_assets_value", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "luxury_assets_value", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "bank_asset_value", ["int64", "float64"]
    ).success == True

    assert pd_df_ge.expect_column_values_to_be_in_type_list(
        "cibil_score", ["int64", "float64"]
    ).success == True

    #assert pd_df_ge.expect_table_column_count_to_equal(23).success == False

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")
  
    return df_validation