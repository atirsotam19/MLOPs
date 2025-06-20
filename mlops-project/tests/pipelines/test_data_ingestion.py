import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.mlops_project.pipelines._02_data_ingestion.nodes import ingestion, build_expectation_suite


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "loan_id": [1, 2],
        "datetime": ["2025-06-01", "2025-06-02"],
        "bank_asset_value": [10000, 15000],
        "cibil_score": [750, 800],
        "loan_approved": [1, 0],
        "commercial_assets_value": [2000, 2500],
        "residential_assets_value": [5000, 6000],
        "income_annum": [30000, 40000],
        "loan_amount": [150000, 200000],
        "luxury_assets_value": [500, 1000],
        "graduate": [1, 0],
        "self_employed": [0, 1],
        "no_of_dependents": [2, 3],
        "loan_term": [10, 15]
    })
    return df


@pytest.fixture
def parameters():
    return {
        "target_column": "loan_approved",
        "to_feature_store": False
    }


def test_ingestion_returns_dataframe(sample_data, parameters):
    df = sample_data
    df = df.rename(columns={"id": "loan_id"})

    df_out = ingestion(df, parameters)
    assert isinstance(df_out, pd.DataFrame)
    assert "datetime" in df_out.columns
    assert "loan_approved" in df_out.columns
    assert "bank_asset_value" in df_out.columns
    assert "graduate" in df_out.columns


@patch("src.mlops_project.pipelines._02_data_ingestion.nodes.to_feature_store")
def test_ingestion_calls_feature_store(mock_store, sample_data, parameters):
    parameters["to_feature_store"] = True
    df = sample_data
    df = df.rename(columns={"id": "loan_id"})

    mock_store.return_value = MagicMock()

    ingestion(df, parameters)

    assert mock_store.call_count == 3


def test_build_expectation_suite_target():
    suite = build_expectation_suite("target_expectations", "target")
    expectations = [e.expectation_type for e in suite.expectations]
    assert "expect_column_distinct_values_to_be_in_set" in expectations


def test_build_expectation_suite_numerical():
    suite = build_expectation_suite("numerical_expectations", "numerical_features")
    types = [e.expectation_type for e in suite.expectations]
    assert "expect_column_values_to_be_of_type" in types
    assert "expect_column_values_to_be_between" in types


def test_build_expectation_suite_categorical():
    suite = build_expectation_suite("categorical_expectations", "categorical_features")
    expectations = [e.expectation_type for e in suite.expectations]
    assert "expect_column_distinct_values_to_be_in_set" in expectations