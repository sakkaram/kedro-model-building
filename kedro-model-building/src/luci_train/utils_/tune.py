import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import optuna
from luci_train.utils_.Focal_classifier import FocalClassifier


# %%
# Define the objective function
def objective_lightgbm(trial, X_train, y_train, categorical_indices):
    """
    Objective function for Optuna hyperparameter optimization for LightGBM.

    Args:
        trial (optuna.trial.Trial): A Trial object that stores the current state of the optimization.
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Training labels.
        categorical_indices (list): List of categorical features indices.

    Returns:
        float: The loss value (negative F1-score) to minimize, ARL value to maximize.
    """

    gamma = trial.suggest_float('gamma', 0, 2.5)
    alpha = trial.suggest_float('alpha', 0.25, 1)

    # Define LightGBM hyperparameters to optimize
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 128, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.75, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.8),
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 5),
        'max_depth': trial.suggest_int('max_depth', 10, 150),
        'verbose': -1,
        'num_boost_round': trial.suggest_int('max_depth', 50, 800),
    }

    # Split the data into train and validation sets
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    _, X_val, _, y_val = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)
    # Initialize the focal loss
    focal_clf = FocalClassifier(alpha=alpha, gamma=gamma, params=params)

    # Fit the model
    focal_clf.fit(X_fit, y_fit)

    # calibrate the model on the calibration set
    focal_clf.calibrate(X_cal, y_cal)

    # Predict on the validation set
    y_pred = focal_clf.predict_proba(X_val)[:, 1]

    y_pred_bin = [int(p > 0.5) for p in y_pred]
    # y_prob = model.predict_proba(X_val)[:, 1]

    # Calculate evaluation metrics
    # ece = expected_calibration_error(y_pred, y_val, M=10)
    precision = metrics.precision_score(y_val, y_pred_bin)
    recall = metrics.recall_score(y_val, y_pred_bin)

    f1 = metrics.f1_score(y_val, y_pred_bin)
    brier_score = metrics.brier_score_loss(y_val, y_pred)
    # arl_value = get_arl(y_prob, 2)
    av_pre = metrics.average_precision_score(y_val, y_pred)

    return f1, av_pre
