""" Reusable preprocessing functions for heart disease prediction. 
All functions return new DataFrames and do not modify the original DataFrame in place. """


import pandas as pd
def handle_missing_values(df, numeric_columns=None, categorical_columns=None):
    """ Handles missing values in the DataFrame. 
    Numeric columns are filled with the median, while categorical columns are filled with the mode.
    Returns a new DataFrame with missing values handled. """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    numeric_columns = numeric_columns or []
    categorical_columns = categorical_columns or []


    missing_cols = [c for c in numeric_columns + categorical_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for col in numeric_columns:
        df[col]= df[col].fillna(df[col].median())
    for col in categorical_columns:
        df[col]= df[col].fillna(df[col].mode()[0])
    return df


def encode_categorical_columns(df, categorical_columns):
    """ Encodes categorical columns using one-hot encoding. 
    Returns a new DataFrame with categorical columns encoded. """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_cols= [c for c in categorical_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The following categorical columns are not present in the DataFrame: {missing_cols}") 

    
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

