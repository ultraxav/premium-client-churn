# libs
from kedro.pipeline import Pipeline, node
from .nodes import reporting

# pipelines
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                reporting,
                inputs=[
                    'trained_model',
                    'model_study',
                    'model_params',
                    'params:split_data',
                ],
                outputs='model_metrics_report',
                name='model_reporting_node',
                tags=['training'],
            ),
        ]
    )
