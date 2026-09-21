"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import handle_missing_values, encode_categorical_columns

@pytest.fixture
def sample_df():
    """Creates a sample DataFrame for testing."""
    return pd.DataFrame({
    'age': [45, 50, np.nan, 60],
    'chol': [200, 210, 190, np.nan],
    'cp': [0, 1, 0, 1],
    'thal': [1, 2, 3, np.nan],
    'target': [0, 1, 0, 1],
})

def test_handle_missing_values_imputes_numeric_with_median(sample_df):
    result= handle_missing_values(sample_df, numeric_columns=['age', 'chol'], categorical_columns=[])
    assert result['age'].isnull().sum() == 0
    assert result['chol'].isnull().sum() == 0
    assert result.loc[2, 'age'] == sample_df['age'].median()

def test_handle_missing_values_imputes_categorical_with_mode(sample_df):
    result = handle_missing_values(sample_df, numeric_columns=[], categorical_columns=["cp", "thal"])
    assert result["cp"].isnull().sum() == 0
    assert result["thal"].isnull().sum() == 0


def test_handle_missing_values_does_not_mutate_original(sample_df):
    original_nulls = sample_df.isnull().sum().sum()
    handle_missing_values(sample_df, numeric_columns=["age", "chol"], categorical_columns=["cp", "thal"])
    assert sample_df.isnull().sum().sum() == original_nulls


def test_handle_missing_values_raises_on_invalid_type():
    with pytest.raises(TypeError):
        handle_missing_values('note a dataframe')

def test_handle_missing_values_raises_on_missing_column(sample_df):
    with pytest.raises(ValueError):
        handle_missing_values(sample_df, numeric_columns=['not_a_column'])

def test_encode_categorical_creates_dummy_columns(sample_df):
    filled = handle_missing_values(sample_df, numeric_columns=[], categorical_columns=['cp', 'thal'])
    result = encode_categorical_columns(filled, categorical_columns=['cp', 'thal'])
    assert 'cp' not in result.columns
    assert any(col.startswith('cp_') for col in result.columns)

def test_encode_categorical_raises_on_invalid_type():
    with pytest.raises(TypeError):
        encode_categorical_columns("not a dataframe", categorical_columns=['cp'])


def test_encode_categorical_raises_on_missing_column(sample_df):
    with pytest.raises(ValueError):
        encode_categorical_columns(sample_df, categorical_columns=["not_a_column"])
