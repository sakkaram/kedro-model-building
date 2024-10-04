import pandas as pd
import numpy as np
import re


def find_date_columns(df, dayfirst=True):
    """
    This function scans the columns of a Pandas dataframe to identify columns that contain dates,
    based on regular expressions, and then returns the names of the columns. It also converts
    the date columns to Pandas datetime format using the appropriate date format.
    """
    date_cols = []

    date_regexes = [
        r'\b\d{4}[./-]\d{2}[./-]\d{2}\b',  # YYYY/MM/DD or YYYY-MM-DD or YYYY.MM.DD
        r'\b\d{2}[./-]\d{2}[./-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
    ]  # define the regular expressions for dates

    for col in df.columns:
        col_data = df[col].astype(str)  # convert the column to string, in case it contains non-string data types
        for i, regex in enumerate(date_regexes):
            if col_data.str.contains(regex).any():  # check if any value in the column matches the date regex
                date_cols.append(col)
                # select the corresponding date format based on the regex index
                if i == 0:
                    year_first = True
                elif i == 1:
                    year_first = False
                df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, yearfirst=year_first,
                                         errors='coerce')  # convert the column to datetime format using the correct
                # format
                break  # break out of the loop if a match is found

    return date_cols, df


def create_age_column(df, birth_date, op_date, age_col_name):
    """
    This function creates an age column from a Pandas dataframe, based on a list of date columns.
    Args: df (Pandas dataframe): Dataframe to create age column from.
        birth_date (str): Name of the birth date column.
        op_date (str): Name of the operation date column.

    Returns: df (Pandas dataframe): Dataframe with age column added.
    """
    df[age_col_name] = df[op_date] - df[birth_date]
    df[age_col_name] = df[age_col_name] / np.timedelta64(1, 'Y')  # convert the age column to years
    return df


def clean_string(input_str):
    """
    This function cleans a string by removing all non-alphanumeric characters and replacing them with spaces.

    Args: input_str (str): String to be cleaned.

    Returns: transformed_string (str): Cleaned string.
    """
    # Perform the transformation on the input string
    transformed_string = re.sub(r'[^A-Za-z0-9_]+', ' ', input_str)
    # Remove leading and trailing spaces
    transformed_string = transformed_string.strip()
    # Replace multiple spaces with a single space
    transformed_string = re.sub(r'\s+', ' ', transformed_string)
    # Convert to lowercase
    transformed_string = transformed_string.lower()

    return transformed_string

