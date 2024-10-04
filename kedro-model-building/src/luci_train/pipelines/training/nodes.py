import json
from typing import  Dict, Any
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, f1_score, average_precision_score, precision_score, recall_score, \
    roc_auc_score
import lightgbm
import luci_train.utils_.data_science as ds
from luci_train.utils_.tune import objective_lightgbm
import luci_train.utils_.plot as plot
from luci_train.utils_.cusum import get_arl
from luci_train.utils_.Focal_classifier import FocalClassifier
from sklearn.model_selection import train_test_split


def tuning(X_train: pd.DataFrame, y_train: pd.Series, categorical_columns: json, parameters: Dict):
    """
    Train a Gradient Boosting Machine.
    Args: pandas DataFrame, pandas Series, parameters dictionary with keys
    'test_size' (float), 'random_state' (int) and 'categorical_columns' (list of strings)

    Returns: Trained Gradient Boosting Machine
    """

    # Get the categorical columns indices
    categorical_columns = categorical_columns['categorical_columns']
    categorical_indices = ds.get_categorical_indices(X_train, categorical_columns)

    # study = optuna.create_study(directions=["maximize", "maximize"])
    # study.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, categorical_indices), n_trials=100)
    #
    # trial_with_max_recall = max(study.best_trials, key=lambda t: t.values[1])
    # params = trial_with_max_recall.params

    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, categorical_indices), n_trials=100)

    trial_with_max_f1 = max(study.best_trials, key=lambda t: t.values[0])
    params = trial_with_max_f1.params

    return params, categorical_indices


def model_training(X_train: pd.DataFrame, y_train: pd.Series, categorical_indices: list, training_params: Dict, parameters: Dict):
    """
    Train a Gradient Boosting Machine.
    Args: pandas DataFrame, pandas Series, parameters dictionary with keys
    'test_size' (float), 'random_state' (int) and 'categorical_columns' (list of strings)

    Returns: Trained Gradient Boosting Machine
    """

    params = training_params
    
    # Used for our FocalClassifier
    alpha = params.pop('alpha')
    gamma = params.pop('gamma')

    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=parameters['calibration_size'],
                                                      random_state=parameters['random_state'])
    # Specifics to Focal Classifier
    focal_clf = FocalClassifier(alpha=alpha, gamma=gamma, params=params)

    focal_clf.fit(X_train, y_train)

    focal_clf.calibrate(X_cal, y_cal, psi_transform=parameters['psi_transform'])


    lightgbm.plot_importance(focal_clf.model, figsize=(12, 16))
    feature_importance = plt.gcf()
    return focal_clf.model, focal_clf.calibrator, feature_importance


def rf_training(X_train: pd.DataFrame, y_train: pd.Series, categorical_columns: json, parameters: Dict):
    """
    Train a Random Forest.
    """

    # Split the data into train and calibration sets
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=parameters['calibration_size'],
                                                      random_state=parameters['random_state'])

    # Get the categorical columns indices
    categorical_columns = categorical_columns['categorical_columns']
    categorical_indices = ds.get_categorical_indices(X_train, categorical_columns)

    # Create the random forest object
    rf = RandomForestClassifier(random_state=parameters['random_state'], class_weight='balanced')

    if parameters['use_smote_boolean']:
        # Create the SMOTE object
        if len(categorical_indices) > 0:
            smote = SMOTENC(categorical_features=categorical_indices, random_state=parameters['random_state'])
        else:
            smote = SMOTE(random_state=parameters['random_state'])

        # Create the pipeline
        clf = Pipeline([('smote', smote), ('rf', rf)])
    else:
        clf = rf

    clf.fit(X_train, y_train)

    # Calibrate the model
    model = CalibratedClassifierCV(clf, cv='prefit')
    model.fit(X_cal, y_cal)

    return model


def evaluate_model(
        model, calibrator, training_params,  X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Dict[str, Any], Any, Any, Any]:
    """Calculates and logs the coefficient of determination.

    Args:
        model: Trained model.
        calibrator: Calibrated model.
        training_params: Parameters for the training. Includes alpha and gamma values for the focal loss.
        X_test: Testing data of independent features.
        y_test: Testing data for target variable.
    """
    alpha = training_params['alpha']
    gamma = training_params['gamma']
    # make predictions for test data
    focal_clf = FocalClassifier(alpha, gamma)
    model = focal_clf.add_trained_model(model, calibrator)
    y_pred = model.predict_proba(X_test)[:, 1]

    # round predictions to first decimal
    y_pred = np.round(y_pred, 2)
    print(y_pred)
    y_binary = y_pred > 0.5
    brier_loss = brier_score_loss(y_test, y_pred)
    f1 = f1_score(y_test, y_binary)
    av_precision = average_precision_score(y_test, y_pred)
    precision = precision_score(y_test, y_binary)
    recall = recall_score(y_test, y_binary)
    roc_auc = roc_auc_score(y_test, y_pred)

    arl = get_arl(y_pred, 2)

    logger = logging.getLogger(__name__)
    logger.info("Model brier loss %.3f on test data.", brier_loss)
    logger.info("Model f1 score %.3f on test data.", f1)
    logger.info("Model average precision %.3f on test data.", av_precision)
    logger.info("Model precision %.3f on test data.", precision)
    logger.info("Model recall %.3f on test data.", recall)
    logger.info("Model roc auc %.3f on test data.", roc_auc)

    metrics = {"ARL": arl, "brier_loss": brier_loss, "f1": f1, "average_precision": av_precision,
               "precision": precision, "recall": recall, "roc_auc": roc_auc}

    distribution_plot, dist_ax = plot.plot_probability_distribution(y_pred)
    roc_plot, roc_ax = plot.plot_roc_curve(y_test, y_pred)
    precision_recall_plot, pre_re_ax = plot.plot_precision_recall_curve(y_test, y_pred)
    return metrics, distribution_plot, roc_plot, precision_recall_plot
