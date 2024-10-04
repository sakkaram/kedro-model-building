"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import delete_nans, encoding, outliers, imputation, split_data, feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=delete_nans,
                inputs=["data_pre_step_3", "params:data_science_options"],
                outputs="data_s_step_1",
                name="delete_nans_node",
            ),
            node(
                func=encoding,
                inputs=["data_s_step_1", "params:data_science_options"],
                outputs=["data_s_step_2", "encoder_dictionary", "categorical_columns"],
                name="encoding_node",
            ),
            node(
                func=outliers,
                inputs=["data_s_step_2", "params:data_science_options", "categorical_columns"],
                outputs=["data_s_step_3", "outlier_remover"],
                name="drop_outliers_node"
            ),
            node(
                func=imputation,
                inputs=["data_s_step_3", "params:data_science_options"],
                outputs=["data_s_step_4", "imputation_kernel", "imputation_plot"],
                name="imputation_node"
            ),
            node(
                func=split_data,
                inputs=["data_s_step_4", "params:data_science_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=feature_selection,
                inputs=["X_train", "y_train", "X_test", "params:data_science_options"],
                outputs=["X_train_fs", "X_test_fs", "selected_features"],
                name="feature_selection_node"
            ),
        ]
    )
