# libs
from kedro.pipeline import Pipeline, node
from .nodes import split_data

# pipelines
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["example_iris_data", "params:example_test_data_ratio"],
                dict(
                    train_x="example_train_x",
                    train_y="example_train_y",
                    test_x="example_test_x",
                    test_y="example_test_y",
                ),
                name="split",
            )
        ]
    )
