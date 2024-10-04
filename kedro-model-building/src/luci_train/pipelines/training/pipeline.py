"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import rf_training, tuning, evaluate_model, model_training


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=tuning,
                inputs=["X_train_fs", "y_train", "categorical_columns", "params:data_science_options"],
                outputs=["training_params", "categorical_indices"],
                name="model_tuning",
            ),
            node(
                func=model_training,
                inputs=["X_train_fs", "y_train", "categorical_indices", "training_params", "params:data_science_options"],
                outputs=["model", "calibrator", "importance_plot"],
                name="model_training",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "calibrator", "training_params", "X_test_fs", "y_test"],
                outputs=["metrics", "probability_dist", "roc_plot", "precision_recall_plot"],
                name="model_evaluation"
            ),
        ]
    )
