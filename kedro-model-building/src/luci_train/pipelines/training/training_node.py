import pandas as pd
import lightgbm
import numpy as np
from scipy import optimize, special
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from typing import Dict


class FocalClassifier:
    def __init__(self, alpha=None, gamma=1.0, params=None):
        self.alpha = alpha
        self.gamma = gamma
        self.params = params
        self.calibrated = False
        self.calibrator = None
        self.model = None

    def fit(self, X, y):
        # initialize focal loss
        focal_loss = FocalLoss(self.gamma, self.alpha)
        # Fit / evaluation data split
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        fit = lightgbm.Dataset(
            X_fit, y_fit,
            init_score=np.full_like(y_fit, focal_loss.init_score(y_fit), dtype=float)
        )

        val = lightgbm.Dataset(
            X_val, y_val,
            init_score=np.full_like(y_val, focal_loss.init_score(y_fit), dtype=float),
            reference=fit
        )
        self.model = lightgbm.train(params=self.params,
                                    train_set=fit,
                                    valid_sets=[fit, val],
                                    valid_names=['fit', 'val'],
                                    fobj=focal_loss.lgb_obj,
                                    feval=focal_loss.lgb_eval)

    def psi_transform(self, X):
        y_positive = special.expit(self.model.predict(X))
        y_negative = 1 - y_positive
        p = np.stack([y_negative, y_positive], axis=1)
        p_cal_unnorm = -1 / (self.gamma * (1 - p) ** (self.gamma - 1) * np.log(p) - ((1 - p) ** self.gamma) / p)
        eta = p_cal_unnorm / np.sum(p_cal_unnorm, axis=1, keepdims=True)
        eta[np.isnan(eta)] = 1.0
        return eta

    def calibrate(self, X, y, psi_transform=True):
        self.calibrator = LR()
        if psi_transform:
            pred = self.psi_transform(X)[:, 1]
        else:
            pred = special.expit(self.model.predict(X))
        self.calibrator.fit(pred.reshape(-1, 1), y)
        self.calibrated = True

    def predict_proba(self, X, calibrated=True, psi_transform=True):
        if calibrated and not self.calibrated:
            raise ValueError('Model not calibrated')

        if psi_transform:
            uncal_pred = self.psi_transform(X)
        else:
            y_positive = special.expit(self.model.predict(X))
            y_negative = 1 - y_positive
            uncal_pred = np.stack([y_negative, y_positive], axis=1)

        if calibrated:
            uncal_pred = uncal_pred[:, 1]
            return self.calibrator.predict_proba(uncal_pred.reshape(-1, 1))
        else:
            return uncal_pred

    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5




def model_training(X_train: pd.DataFrame, y_train: pd.Series, categorical_indices: list, training_params: Dict, parameters: Dict):
    """
    Train a Gradient Boosting Machine.
    Args: pandas DataFrame, pandas Series, parameters dictionary with keys
    'test_size' (float), 'random_state' (int) and 'categorical_columns' (list of strings)

    Returns: Trained Gradient Boosting Machine
    """

    params = training_params

    alpha = params.pop('alpha')
    gamma = params.pop('gamma')

    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=parameters['calibration_size'],
                                                      random_state=parameters['random_state'])

    focal_clf = FocalClassifier(alpha=alpha, gamma=gamma, params=params)

    focal_clf.fit(X_train, y_train)

    focal_clf.calibrate(X_cal, y_cal, psi_transform=parameters['psi_transform'])
    lightgbm.plot_importance(focal_clf.model, figsize=(12, 16))
    feature_importance = plt.gcf()
    return focal_clf.model, feature_importance
