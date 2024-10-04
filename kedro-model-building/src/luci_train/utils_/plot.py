import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


def plot_probability_distribution(y_proba):
    """
    A function to plot the probability distribution of the predicted probabilities.

    Parameters
    ----------
    y_proba : array-like of shape (n_samples,)
        Target scores or probability estimates for the positive class.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The matplotlib Figure object containing the plotted figure.

    ax : matplotlib.axes._subplots.AxesSubplot object
        The matplotlib AxesSubplot object containing the plotted axes.
    """

    # Set style and context
    sns.set_style("white")
    sns.set_context("talk")

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histogram using seaborn's histplot function
    sns.histplot(y_proba, color='#000000', kde=False)

    # Set the title, xlabel, and ylabel with Helvetica font
    ax.set_title('Probability of Complication', fontsize=12, fontfamily='Helvetica')
    ax.set_xlabel('Probability', fontsize=10, fontfamily='Helvetica')
    ax.set_ylabel('Frequency', fontsize=10, fontfamily='Helvetica')

    # Set the x-axis limits
    ax.set_xlim(0, 1)

    # Make the x and y-axis thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    # Remove the top and right spines from the plot
    sns.despine()

    return fig, ax


def plot_roc_curve(y_true, y_proba):
    """
    A function to plot ROC curve and compute ROC-AUC score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.

    y_proba : array-like of shape (n_samples,)
        Target scores or probability estimates for the positive class.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The matplotlib Figure object containing the plotted figure.

    ax : matplotlib.axes._subplots.AxesSubplot object
        The matplotlib AxesSubplot object containing the plotted axes.
    """
    # Compute false positive rate, true positive rate and thresholds for ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(y_true, y_proba)

    # Define colors
    color1 = '#000000'  # black
    color2 = '#FF0000'  # red
    color3 = '#008000'  # green

    # Set style and context
    sns.set_style("white")
    sns.set_context("talk")

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, label='ROC-AUC (area = {:.3f})'.format(roc_auc), color=color1, linewidth=1.2)
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random Model', color=color2, linewidth=1)
    ax.plot([0, 0], [0, 1], linestyle='--', color=color3, label='Perfect Model', linewidth=1)
    ax.plot([0, 1], [1, 1], linestyle='--', color=color3, linewidth=1)

    # Set x and y-axis labels and title with Helvetica font
    ax.set_xlabel('False Positive Rate', fontsize=10, fontfamily='Helvetica')
    ax.set_ylabel('True Positive Rate', fontsize=10, fontfamily='Helvetica')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontfamily='Helvetica')

    # Set legend with Helvetica font
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(0.5)

    # Set x and y-axis limits with buffer
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.01])

    # Make the x and y-axis thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

    sns.despine()

    return fig, ax



def plot_precision_recall_curve(y_true, y_proba):
    """
    A function to plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.

    y_proba : array-like of shape (n_samples,)
        Target scores or probability estimates for the positive class.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The matplotlib Figure object containing the plotted figure.

    ax : matplotlib.axes._subplots.AxesSubplot object
        The matplotlib AxesSubplot object containing the plotted axes.
    """
    # Calculate average precision score
    avg_precision = average_precision_score(y_true, y_proba)

    # Compute precision, recall and thresholds for precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    counts = y_true.value_counts()
    counts_0 = counts[0]
    counts_1 = counts[1]
    random_model_line = counts_1 / (counts_0 + counts_1)

    # Set style and context
    sns.set_style("white")
    sns.set_context("talk")

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors
    color1 = '#000000'  # black
    color2 = '#FF0000'  # red
    color3 = '#008000'  # green

    ax.plot(recall, precision, label='Average Precision (area = {:.3f})'.format(avg_precision), color=color1,
            linewidth=1.2)
    ax.plot([0, 1], [1, 1], linestyle='--', color=color3, label='Perfect Model', linewidth=1)
    ax.plot([0, 1], [random_model_line, random_model_line], linestyle='--', color=color2, label='Random Model',
            linewidth=1)
    ax.set_xlabel('Recall', fontsize=10, fontfamily='Helvetica')
    ax.set_ylabel('Precision', fontsize=10, fontfamily='Helvetica')
    ax.set_title('Precision-Recall Curve', fontsize=12, fontfamily='Helvetica')

    # Set legend with Helvetica font
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(0.5)

    # Set x and y-axis limits with buffer
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.01])

    # Make the x and y-axis thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

    sns.despine()

    return fig, ax


def cusum_plot(dataframe, groups=None, cl=2, cusum_type='complications'):
    """
    Plots a CUSUM (cumulative sum) chart for either the accumulation of complications or successes.

    Args:
    - dataframe (pandas.DataFrame): The data to be plotted, which must contain 'Patient_no' and 'CUSUM' columns.
    - groups (list of tuples): The groups of patients to be highlighted in the plot, where each tuple represents the
                               start and end patient numbers for a group.
    - cl (float): The threshold value for the CUSUM chart. Defaults to 2.
    - cusum_type (str): The type of CUSUM chart to be plotted, either 'complications' or 'survival'. Defaults to
                        'complications'.

    Returns:
    - plt (matplotlib.pyplot object): The matplotlib.pyplot object containing the plotted figure.

    Raises:
    - ValueError: If `cusum_type` is not either 'complications' or 'survival'.
    - ValueError: If groups are not provided as a list of tuples.
    - ValueError: If save_fig_path is not a valid directory.

    """

    # Set style and context
    sns.set_style("white")
    sns.set_context("talk")

    # Define colors
    color1 = '#000000'  # black
    color2 = '#FF0000'  # red
    color3 = '#008000'  # green
    plt.figure(figsize=(12, 6))
    if cusum_type == 'complications':
        plt.plot(dataframe['Patient_no'], dataframe['CUSUM'], color=color1, linewidth=1.2)
        group_color = color2
        y_limit = max(max(dataframe['CUSUM']) + 1, cl + 0.5)
        plt.axhline(y=cl, linestyle='--', color=color2, alpha=0.7, linewidth=1, label='Threshold')
        group_label = 'Patients that lead the monitoring out-of-control'
        plot_title = 'CUSUM plot for accumulation of complications'

    elif cusum_type == 'survival':
        plt.plot(dataframe['Patient_no'], dataframe['CUSUM'], color=color1, linewidth=1.2)
        group_color = color3
        y_limit = min(min(dataframe['CUSUM']) - 1, -cl - 0.5)
        plt.axhline(y=-cl, linestyle='--', color=color3, alpha=0.7, linewidth=1.2, label='Threshold')
        group_label = None
        plt.title('', fontsize=12, fontfamily='Helvetica')
        plot_title = 'CUSUM plot for accumulation of successes'

    else:
        raise ValueError('Invalid cusum_type. Must be either "complications" or "survival".')

    # Add colored lines for each group
    if groups is not None:
        for group in groups:
            if len(group) != 2:
                raise ValueError(
                    'Invalid group format. Each group should be a tuple of length 2 representing the start '
                    'and end patient numbers.')
            start, end = group
            group_data = dataframe[(dataframe['Patient_no'] >= start) & (dataframe['Patient_no'] <= end)]
            plt.plot(group_data['Patient_no'], group_data['CUSUM'], color=group_color, linewidth=1.2, label=group_label)

    plt.title(plot_title, fontsize=12, fontfamily='Helvetica')
    plt.xlabel('Patient Number', fontsize=10, fontfamily='Helvetica')
    plt.ylabel('CUSUM Value', fontsize=10, fontfamily='Helvetica')

    # Set legend
    legend = plt.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(0.5)

    # Set x and y-axis limits
    plt.ylim([0, y_limit])
    plt.xlim([0, max(dataframe['Patient_no']) + 1])

    # Make the x and y-axis thinner
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    sns.despine()

    return plt
