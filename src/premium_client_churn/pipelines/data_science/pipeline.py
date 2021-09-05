# libs
from kedro.pipeline import Pipeline, node
from .nodes import predict, report_accuracy, train_model

# pipelines
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["example_train_x", "example_train_y", "parameters"],
                "example_model",
                name="train",
            ),
            node(
                predict,
                dict(model="example_model", test_x="example_test_x"),
                "example_predictions",
                name="predict",
            ),
            node(
                report_accuracy,
                ["example_predictions", "example_test_y"],
                None,
                name="report",
            ),
        ]
    )
