"""Data validation tests the heart disease dataset."""

import pandas as pd
import pytest


DATA_PATH= "data/raw/heart.csv"

EXPECTED_COLUMNS= {
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    }

@pytest.fixture(scope='module')
def raw_df():
    return pd.read_csv(DATA_PATH)


def test_expected_columns_present(raw_df):
    assert EXPECTED_COLUMNS.issubset(set(raw_df.columns))


def test_target__contains_only_expected_values(raw_df):
    assert set(raw_df['target'].dropna().unique()).issubset({0, 1})


def test_numeric_features_within_expected_ranges(raw_df):
    assert raw_df['age'].dropna().between(0, 120).all()
    assert raw_df['trestbps'].dropna().between(0, 300).all()
    assert raw_df['chol'].dropna().between(0, 800).all()
    assert raw_df['thalach'].dropna().between(0, 250).all()



def test_minimum_row_and_feature_count(raw_df):
    assert len(raw_df) >= 1000
    assert len(raw_df.columns) >= 8

