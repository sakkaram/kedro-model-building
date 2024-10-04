import json
from typing import Tuple, Dict, Any

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import luci_train.utils_.data_science as ds
import luci_train.utils_.data_handling as dh
from luci_train.utils_.feature_selection import luciusRFECV, BorutaPy, BorutaPyForLGB
from lightgbm import LGBMClassifier


def delete_nans(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Delete rows and columns with high percentage of NaNs. Delete rows with NaNs in target column.

     Args: pandas DataFrame,
    parameters dictionary with keys 'row_threshold' and 'column_threshold' (floats) and 'target_column' (string or int)

    Returns: pandas DataFrame with rows and columns with high percentage of NaNs deleted and rows with NaNs in target column deleted
    """
    df_copy = df.copy()
    row_threshold = parameters['row_threshold']
    column_threshold = parameters['column_threshold']
    target_column = dh.clean_string(parameters['target_variable'])

    # Delete rows with NaNs in target column
    df_copy = df_copy.dropna(subset=[target_column])

    # Delete rows with high percentage of NaNs
    df_copy = df_copy.dropna(thresh=int(row_threshold * len(df_copy.columns)), axis=0)

    # Delete columns with high percentage of NaNs
    df_copy = df_copy.dropna(thresh=int(column_threshold * len(df_copy)), axis=1)

    return df_copy


def encoding(df: pd.DataFrame, parameters: Dict) -> Tuple[DataFrame, Any, Any]:
    """
    Encode categorical columns

    Args: pandas DataFrame,
    parameters dictionary with key 'categorical_columns' (list of strings)

    Returns: pandas DataFrame with categorical columns encoded
    """
    df_copy = df.copy()

    target_column = dh.clean_string(parameters['target_variable'])
    positive_class = parameters['positive_class']
    negative_class = parameters['negative_class']
    if type(positive_class) == str:
        positive_class = dh.clean_string(positive_class)
    if type(negative_class) == str:
        negative_class = dh.clean_string(negative_class)

    # Map positive and negative class to 0 and 1
    if target_column in df_copy.columns:
        df_copy[target_column] = df_copy[target_column].map({negative_class: 0, positive_class: 1})
    else:
        raise ValueError('Target column not found in dataframe')

    # Transform categorical columns to match the preprocessed data
    if parameters['categorical_columns']:
        categorical_columns = [dh.clean_string(col) for col in parameters['categorical_columns']]
    else:
        categorical_columns = []

    # Find categorical columns if is asked
    if parameters['find_categorical_columns_boolean']:
        categorical_columns = ds.detect_categorical_columns(df_copy, parameters['cardinality_threshold'],
                                                            parameters['numerical_columns'])
        categorical_columns.remove(target_column)

    # Encode categorical columns
    encoded_df, encoding_dict = ds.encode_categorical_columns(df_copy, categorical_columns)

    # Make categorical columns of type category
    encoded_df[categorical_columns] = encoded_df[categorical_columns].astype('category')

    return encoded_df, encoding_dict, {'categorical_columns': categorical_columns}


def outliers(df: pd.DataFrame, parameters: Dict, categorical_columns: json) -> Tuple[pd.DataFrame, Any]:
    """
    Delete outliers .

    Args: pandas DataFrame, parameters dictionary with keys 'outlier_threshold' (float).

    Returns: pandas DataFrame with outliers deleted and the outlier kernel used.
    """
    df_copy = df.copy()
    if parameters['drop_outliers_boolean']:

        outlier_threshold = parameters['outlier_threshold']
        categorical_columns = categorical_columns['categorical_columns']

        # Create an instance of the OutlierRemover class
        outlier_remover = ds.OutlierRemover(outlier_threshold, categorical_columns)

        # Fit and transform the dataframe
        df_copy = outlier_remover.fit_transform(df_copy)
    else:
        outlier_remover = False

    return df_copy, outlier_remover


def imputation(df: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Impute missing values.

    Args: pandas DataFrame.

    Returns: pandas DataFrame with imputed missing values and the imputer used.
    """
    df_copy = df.copy()

    # Create an instance of the Imputer class
    df_copy, imputation_kernel, plot = ds.imputation_complete(df_copy, random_state=parameters['random_state'])
    return df_copy, imputation_kernel, plot


def split_data(df: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into train and test sets.

    Args: pandas DataFrame, parameters dictionary with keys 'test_size' (float) and 'random_state' (int).

    Returns: pandas DataFrames with train and test sets.
    """
    df_copy = df.copy()

    target_column = dh.clean_string(parameters['target_variable'])

    X = df_copy.drop(target_column, axis=1)
    y = df_copy[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'],
                                                        random_state=parameters['random_state'])

    return X_train, X_test, y_train, y_test


def feature_selection(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, parameters: Dict) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict]:
    """
    Select features, with Boruta and RFE.

    Args: pandas DataFrames with train and test sets, parameters dictionary with keys 'boruta_selection_boolean',
    'random_state', 'min_features' and 'rfe_selection_boolean'

    Returns: pandas DataFrames with train and test sets with selected features and the selected features
    """
    method = parameters['method']

    if parameters['boruta_selection_boolean']:
        # initialize Boruta
        if method == 'lgbm':
            model = LGBMClassifier(num_boost_round=100, class_weight='balanced', random_state=parameters['random_state'])
            boruta = BorutaPyForLGB(max_iter=100, estimator=model, n_estimators='auto', verbose=10,
                                    random_state=parameters['random_state'])
        elif method == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=parameters['random_state'])
            boruta = BorutaPy(max_iter=100, estimator=model, n_estimators='auto', verbose=10,
                              random_state=parameters['random_state'])
        else:
            raise ValueError('Method not supported')

        # fit Boruta (it accepts np.array, not pd.DataFrame)
        boruta.fit(X_train, y_train)

        if boruta.n_features_ >= parameters['min_features']:
            selected = X_train.columns[boruta.ranking_ == 1].tolist()
            X_train_fs = X_train[selected]
            X_test_fs = X_test[selected]
        else:
            rank = 2
            while len(X_train.columns[boruta.ranking_ < rank].tolist()) < parameters['min_features']:
                rank += 1
            selected = X_train.columns[boruta.ranking_ < rank].tolist()
            X_train_fs = X_train[selected]
            X_test_fs = X_test[selected]
    else:
        selected = X_train.columns.tolist()
        X_train_fs = X_train
        X_test_fs = X_test

    if parameters['rfecv_boolean']:
        if len(selected) > parameters['min_features']:
            rfe = luciusRFECV(method=method, random_state=parameters['random_state'], metric=parameters['main_metric'])
            rfe = rfe.fit(X_train_fs, y_train)

            if rfe.n_features >= parameters['min_features']:
                X_train_fs = rfe.transform_X(X_train_fs)
                X_test_fs = rfe.transform_X(X_test_fs)
                selected = rfe.selected_features

    return X_train_fs, X_test_fs, {'features': selected}
