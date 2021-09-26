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
                    'train_data',
                    'valid_data',
                    'test_data',
                    'leader_data',
                    'model_metrics',
                    'trained_model',
                    'model_study',
                    'model_params',
                ],
                outputs='model_metrics_report',
                name='model_reporting_node',
                tags=['training'],
            ),
        ]
    )
