""" One-time script to prepare the data for the heart prediction model. 
Injecting synthetic missing values into the raw Kaggle heart disease dataset. 
Run this as a manual step as it is not part of the recurring pipeline. 
Re-Running this will stack additional missing values on top of the existing ones."""


import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/raw/heart.csv"
RANDOM_SEED = 12345
MISSING_Rate= 0.05

# Mix of Numeric and Categorical features

COLUMNS_TO_CORRUPT= ['chol', 'trestbps', 'thalach', 'oldpeak', 'age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

def inject_missing_values(df, columns, missing_rate, seed):
    """ Injects missing values into the specified columns of the DataFrame at the given missing rate.
    The random seed is used for reproducibility. """
    rng = np.random.default_rng(seed)
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for col in columns:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def main():

    df = pd.read_csv(RAW_DATA_PATH)
    existing_nulls = df.isnull().sum().sum()
    if existing_nulls > 0:
        print(f"Warning: The dataset already contains {existing_nulls} missing values. "
              f"Re-running this script will add more missing values on top of the existing ones.")

    df = inject_missing_values(df, COLUMNS_TO_CORRUPT, MISSING_Rate, RANDOM_SEED)

    df.to_csv(RAW_DATA_PATH, index=False)
    print("Missing values injected per column:")
    print(df[COLUMNS_TO_CORRUPT].isnull().sum())
    print(f'\nTotal Missing : {df.isnull().sum().sum()} / {df.size} cells')


if __name__ == "__main__":
    main()
    