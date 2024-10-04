from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import miceforest as mf
from matplotlib import pyplot as plt


def detect_categorical_columns(df: pd.DataFrame, threshold: float = 0.05, exclude_columns: list = None) -> list:
    """
    Detect categorical columns in a pandas DataFrame based on the data type, cardinality, and an exclusion list.

    Args:
        df: pandas DataFrame.
        threshold: Cardinality threshold (percentage) to consider a column as categorical.
                   Default is set to 0.05 (5%).
        exclude_columns: List of column names to exclude from being considered as categorical. Default is None.

    Returns:
        List of column names identified as categorical.

    """
    categorical_columns = []
    total_rows = len(df)

    for col in df.columns:
        if exclude_columns and col in exclude_columns:
            continue

        col_type = df[col].dtype

        if col_type == "object" or pd.api.types.is_categorical_dtype(col_type):
            categorical_columns.append(col)
        else:
            unique_values = df[col].nunique()
            cardinality = unique_values / total_rows

            if cardinality <= threshold:
                categorical_columns.append(col)

    return categorical_columns


def encode_categorical_columns(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, dict[Any, dict]]:
    """
    Encode categorical columns in a pandas DataFrame.

    Args: pandas DataFrame, list of column names to encode

    Returns: pandas DataFrame with categorical columns encoded, dictionary with encoding mappings
    """
    encoded_df = df.copy()
    encoding_dict = {}

    for col in columns:
        unique_values = df[col].dropna().unique()
        mapping = {val: i for i, val in enumerate(unique_values)}
        encoding_dict[col] = mapping
        encoded_df[col] = df[col].map(mapping).astype('float64')

    return encoded_df, encoding_dict


class OutlierRemover:
    def __init__(self, k=3, categorical_columns=None):
        self.k = k
        self.categorical_columns = categorical_columns
        self.bounds = {}

    def fit_transform(self, df):
        df_filtered = df.copy()
        self.bounds = {}

        for column_name in df.columns:
            if column_name not in self.categorical_columns:
                # Calculate the median and median absolute deviation
                median = df_filtered[column_name].median()
                mad = (df_filtered[column_name] - df_filtered[column_name].mean()).abs().mean()
                # Calculate the lower and upper bounds using the median and MAD
                lower_bound = median - self.k * mad
                upper_bound = median + self.k * mad

                # Remove values where the value in the specified column is outside the bounds
                df_filtered[column_name] = df_filtered[column_name][
                    ~((df_filtered[column_name] < lower_bound) | (df_filtered[column_name] > upper_bound))]

                # Save the lower and upper bounds for the column
                self.bounds[column_name] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

        return df_filtered

    def transform(self, df):
        df_filtered = df.copy()

        for column_name, bounds in self.bounds.items():
            lower_bound = bounds['lower_bound']
            upper_bound = bounds['upper_bound']

            # Remove values where the value in the specified column is outside the saved bounds
            df_filtered[column_name] = df_filtered[column_name][
                ~((df_filtered[column_name] < lower_bound) | (df_filtered[column_name] > upper_bound))]

        return df_filtered


def imputation_complete(df, n_iterations=3, kernel=None, random_state=42):
    """
    Imputes dataset.
    :param df: Pandas Dataframe of the dataset
    :param n_iterations: Number of iterations of the imputation process, default is 3 iterations
    :param kernel: mice ImputationKernel, default is None
    :param random_state: random state, default is 42
    :return: imputed dataset, mice ImputationKernel

    """
    print(f"Data shape is {df.shape}\n")

    if kernel is None:
        kernel = mf.ImputationKernel(df, save_models=5, random_state=random_state)
        kernel.mice(n_iterations)
        imp_df = kernel.complete_data(inplace=False)
    else:
        df_imputation = kernel.impute_new_data(new_data=df, random_state=random_state)
        imp_df = df_imputation.complete_data(0)
    n_nan = imp_df.isna().sum(axis=0).sum()
    print(f"Data after imputation have {n_nan} NaN values\n")
    if n_nan > 10:
        raise Exception('Imputation was not executed correctly')
    imp_df = imp_df.dropna()
    print(f"Data after imputation, shape {imp_df.shape}\n")

    kernel.plot_imputed_distributions(wspace=1, hspace=1, top=3, left=0, right=3)

    dist_plot = plt.gcf()
    # Adjust the figure size
    dist_plot.set_size_inches(12, 12)  # Adjust the size according to your preference

    # Automatically adjust the subplot parameters
    plt.tight_layout()

    return imp_df, kernel, dist_plot


def get_categorical_indices(df, categorical_columns):
    """
    Returns the indices of the categorical columns
    :param df: Pandas Dataframe of the dataset
    :param categorical_columns: list of categorical columns
    :return: list of indices of the categorical columns
    """
    # Get the names of the selected categorical features
    categorical_selected = [x for x in categorical_columns if x in df.columns.tolist()]

    # Initialize a list to store the column indices
    column_indices = []

    categorical_indices = []
    for column in categorical_selected:
        categorical_indices.append(df.columns.get_loc(column))
    return categorical_indices
