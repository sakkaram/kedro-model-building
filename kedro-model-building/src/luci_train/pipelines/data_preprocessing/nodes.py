from typing import Tuple, Dict
import luci_train.utils_.data_handling as dh
import pandas as pd


def delete_columns_set_index(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Delete columns from a pandas DataFrame. And set the index to the column specified in the parameters.

    Args:
    - df (pandas DataFrame): DataFrame to delete columns from.
    - parameters (dict): Dictionary of parameters for the node.

    Returns:
    - df (pandas DataFrame): DataFrame with columns deleted.
    """

    # Make a copy of the DataFrame so that the original is not modified.
    df_copy = df.copy()
    # Set the index to the column specified in the parameters.
    if parameters['set_index_boolean']:
        df_copy = df_copy.drop_duplicates(subset=[parameters['index_column']], keep='first')
        df_copy = df_copy.set_index(parameters['index_column'])
    # Delete columns specified in the parameters.
    if parameters['drop_boolean']:
        df_copy = df_copy.drop(parameters['columns_to_drop'], axis=1)
    return df_copy


def handle_date_columns(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Handle date columns in a pandas DataFrame.

    Args:
    - df (pandas DataFrame): DataFrame to handle date columns in.
    - parameters (dict): Dictionary of parameters for the node.

    Returns:
    - df (pandas DataFrame): DataFrame with date columns handled.
    """

    # Make a copy of the DataFrame so that the original is not modified.
    df_copy = df.copy()

    # Convert date columns to datetime format.
    date_cols, df_copy = dh.find_date_columns(df_copy, parameters['date_dayfirst'])

    if parameters['age_exists'] is False:
        # Create age column from date columns.
        df_copy = dh.create_age_column(df_copy, parameters['birth_date_col'], parameters['op_date_col'], 'Age')

    if parameters['op_date_col']:
        # Calculating average patients per year
        n_patients_per_year = df_copy[parameters['op_date_col']].groupby(df_copy[parameters['op_date_col']].dt.year).agg(
            'count').mean()
    else:
        n_patients_per_year = -1.0
    # Delete date columns.
    df_copy = df_copy.drop(date_cols, axis=1)
    print(n_patients_per_year)
    return df_copy, {"n_patients_per_year": n_patients_per_year}


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean strings in a pandas DataFrame.

    Args:
    - df (pandas DataFrame): DataFrame to clean strings in.

    Returns:
    - df_copy (pandas DataFrame): DataFrame with strings cleaned.
    """

    # Make a copy of the DataFrame so that the original is not modified.
    df_copy = df.copy()

    # Transform column names using transform_string function
    new_names = {col: dh.clean_string(col) for col in df.columns}
    df_copy = df.rename(columns=new_names)

    # Clean strings in records
    df_copy = df_copy.applymap(lambda x: dh.clean_string(x) if isinstance(x, str) else x)

    return df_copy

