import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.mlops_project.pipelines._02_data_ingestion.nodes import ingestion, build_expectation_suite


@pytest.fixture
def sample_data():
    df1 = pd.DataFrame({
        "id": [1, 2],
        "bank_asset_value": [10000, 15000],
        "cibil_score": [750, 800],
        "month": ["jan", "feb"],
        "loan_approved": [1, 0],
    })

    df2 = pd.DataFrame({
        "id": [1, 2],
        "graduate": [1, 0],
        "loan_term": [10, 15],
        "no_of_dependents": [2, 1],
        "self_employed": [0, 1]
    })

    return df1, df2


@pytest.fixture
def parameters():
    return {
        "target_column": "loan_approved",
        "to_feature_store": False
    }


def test_ingestion_returns_dataframe(sample_data, parameters):
    df1, df2 = sample_data
    df1 = df1.rename(columns={"id": "index"})
    df2 = df2.rename(columns={"id": "index"})

    df_out = ingestion(df1, df2, parameters)
    assert isinstance(df_out, pd.DataFrame)
    assert "datetime" in df_out.columns
    assert "loan_approved" in df_out.columns
    assert "bank_asset_value" in df_out.columns
    assert "graduate" in df_out.columns


@patch("src.your_project_name.pipelines.data_engineering.nodes.to_feature_store")
def test_ingestion_calls_feature_store(mock_store, sample_data, parameters):
    parameters["to_feature_store"] = True
    df1, df2 = sample_data
    df1 = df1.rename(columns={"id": "index"})
    df2 = df2.rename(columns={"id": "index"})

    mock_store.return_value = MagicMock()

    ingestion(df1, df2, parameters)

    assert mock_store.call_count == 3


def test_build_expectation_suite_target():
    suite = build_expectation_suite("target_expectations", "target")
    expectations = [e.expectation_type for e in suite.expectations]
    assert "expect_column_distinct_values_to_be_in_set" in expectations


def test_build_expectation_suite_numerical():
    suite = build_expectation_suite("numerical_expectations", "numerical_features")
    types = [e.expectation_type for e in suite.expectations]
    assert "expect_column_values_to_be_of_type" in types
    assert "expect_column_min_to_be_between" in types


def test_build_expectation_suite_categorical():
    suite = build_expectation_suite("categorical_expectations", "categorical_features")
    expectations = [e.expectation_type for e in suite.expectations]
    assert "expect_column_distinct_values_to_be_in_set" in expectations