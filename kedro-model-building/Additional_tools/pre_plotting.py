import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def pre_plotting_CUSUM(prob, y_test, cusum_type='complications'):
    """
    Calculates the values for CUSUM plot
    :param prob: probability of negative outcome
    :param y_test: target of test data
    :param cusum_type: (str, optional): the type of cusum to be included in the reordered dataframe.
          Valid options are 'complications' and 'success'. Defaults to 'complications'.

    :return: pandas dataframe containing x,y-axis for CUSUM plot
    """

    # R1=2 higher values on CUSUM indicate more complications than expected,
    # R1 = 0.5 higher values on CUSUM indicate fewer complications than expected

    cusum_col_name = 'CUSUM'

    if cusum_type == 'complications':
        R1 = 2
    elif cusum_type == 'success':
        R1 = 0.5
    else:
        raise ValueError("Invalid cusum_type. Valid options are 'complications' and 'success'.")

    pred_out = np.stack((prob, y_test), axis=1)
    x = []
    y = []
    cup = 0
    for index in range(len(pred_out)):
        prediction = pred_out[index, 0]  # surgical risk for patient in index
        outcome = pred_out[index, 1]  # surgical outcome for patient in index
        # negative outcome
        if outcome == 1:
            w = math.log(R1 / (1 - prediction + R1 * prediction))
        # positive outcome
        else:
            w = math.log(1 / (1 - prediction + R1 * prediction))
        # update CUSUM value
        cup = max(0, cup + w)
        # store patient number and CUSUM value
        x.append(index + 1)
        y.append(cup)
    plot_data = {'Patient_no': x, cusum_col_name: y}
    plot_df = pd.DataFrame(plot_data)
    if R1 == 0.5:
        plot_df[cusum_col_name] = (-1) * plot_df[cusum_col_name]
    return plot_df

