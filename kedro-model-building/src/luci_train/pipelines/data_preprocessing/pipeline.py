from kedro.pipeline import Pipeline, node, pipeline

from .nodes import delete_columns_set_index, handle_date_columns, clean_strings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=delete_columns_set_index,
                inputs=["raw_data", "params:data_options"],
                outputs="data_pre_step_1",
                name="delete_columns_set_index_node",
            ),
            node(
                func=handle_date_columns,
                inputs=["data_pre_step_1", "params:date_options"],
                outputs=["data_pre_step_2","n_patients"],
                name="handle_date_columns_node"
            ),
            node(
                func=clean_strings,
                inputs="data_pre_step_2",
                outputs="data_pre_step_3",
                name="clean_strings_node",
            ),
        ]
    )

