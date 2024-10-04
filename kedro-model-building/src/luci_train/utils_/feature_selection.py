"""
Author: Daniel Homola <dani.homola@gmail.com>
Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
License: BSD 3 clause
"""

from __future__ import print_function, division
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier




class luciusRFECV:
    """
        A class for feature selection and training a classifier using recursive feature elimination
        with cross-validation (RFECV).

            Parameters
    ----------
    method : str
        The type of classifier to use. Must be one of 'rf' (random forest) or 'cb' (CatBoost).
    metric : str, optional
        The evaluation metric to use for feature selection and model evaluation. Default is 'average_precision'.

    Attributes:
    -----------
    feature_importance : array-like of shape (n_features,)
        The feature importances ranked in descending order.

    cv_mean_scores : array-like of shape (n_features,)
        The mean cross-validation scores of the selected features.

    f_importances : array-like of shape (n_features,)
        The feature importances of the selected features.

    get_sup : object
        The fitted estimator object with support for feature selection (i.e. it has an attribute named 'support_').

    n_features : int
        The number of selected features.

    selected_features : array-like of shape (n_features,)
        The indices of the selected features.

    """

    def __init__(self, method, random_state=42, metric='average_precision'):
        """
                Initialize the luciusRFECV object.

        Parameters
        ----------

        metric : str, optional
            The evaluation metric to use for feature selection and model evaluation. Default is 'average_precision'.
        """
        self.feature_importance = None
        self.cv_mean_scores = None
        self.f_importances = None
        self.get_sup = None
        self.n_features = None
        self.selected_features = None
        self.method = method
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y):
        """
                Fit the model to the training data.

                Parameters
                ----------
                X : pandas DataFrame
                    The training data.
                y : pandas Series or array-like
                    The target labels.

                Returns
                -------
                self : luciusRFECV object
                    The fit model.
                """
        if self.method == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1)
        elif self.method == 'lgbm':
            model = LGBMClassifier(random_state=self.random_state, class_weight='balanced')
        else:
            raise ValueError("The method must be one of 'rf' or 'lgbm'.")
        if len(X) < 150:
            n_folds = 3
        else:
            n_folds = 5

        # Run recursive feature elimination with cross-validation (RFECV)
        sel = RFECV(model, scoring=self.metric, cv=n_folds)
        sel.fit(X, y)
        # Store the number of selected features
        self.n_features = sel.n_features_

        # If the number of selected features is less than 3, set it to 3
        if self.n_features < 3:
            self.n_features = 3

        # Store the mask of selected features
        self.get_sup = sel.get_support()

        # Store the feature importances
        self.f_importances = sel.estimator_.feature_importances_

        # Store the mean cross-validation scores
        self.cv_mean_scores = sel.cv_results_.get('mean_test_score')

        # Create a list of tuples of 'feature name' : importance, sorted by importance in descending order
        self.feature_importance = sorted(list(zip(X.columns[self.get_sup], self.f_importances)),
                                         key=lambda x: x[1],
                                         reverse=True)

        # Store the selected features
        self.selected_features = [x[0] for x in self.feature_importance[:self.n_features]]

        return self

    def transform_X(self, X):
        """
        Transform the input data by selecting the features identified in the RFE-CV process.

        Parameters
        ----------
        X : pandas DataFrame
            The input data.

        Returns
        -------
        X_transformed : pandas DataFrame
            The transformed input data with only the selected features.
        """

        if self.n_features is None or self.feature_importance is None:
            raise Exception("luciusRFECV has to be fitted, before using any other method. Use fit.")
        else:
            X_rfe = X[self.selected_features]
        return X_rfe

    def plot_CV_scores(self):
        if self.cv_mean_scores is None:
            raise Exception("luciusRFECV has to be fitted, before using any other method. Use fit.")
        else:
            # Create a Pandas dataframe with the data for plotting
            df = pd.DataFrame({'num_features': range(1, len(self.cv_mean_scores) + 1),
                               'cv_score': self.cv_mean_scores})
            return df

    def categorical_selected_indices(self, transformed_df, categorical_cols):
        """
        Get the indices of the selected categorical features in the transformed dataframe.

        Parameters
        ----------
        transformed_df : pandas DataFrame
            The transformed dataframe with only the selected features.
        categorical_cols : list of str
            The names of the categorical columns in the original data.

        Returns
        -------
        column_indices : list of int
            The indices of the selected categorical features in the transformed dataframe.
        """
        # Get the names of the selected categorical features
        categorical_selected = [x for x in categorical_cols if x in self.selected_features]

        # Initialize a list to store the column indices
        column_indices = []

        # Iterate through the selected categorical features and get the indices in the transformed dataframe
        for column_name in categorical_selected:
            column_index = transformed_df.columns.get_loc(column_name)
            column_indices.append(column_index)

        # Return the list of column indices
        return column_indices
