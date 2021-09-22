# libs
from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, predict

# pipeline
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                inputs=['feature_data', 'params:data_science'],
                outputs=['train_data', 'valid_data', 'test_data', 'leader_data'],
                name='split_data_node',
            ),
            node(
                train_model,
                inputs=['train_data', 'valid_data', 'params:data_science'],
                outputs=['trained_model', 'model_study', 'model_params'],
                name='train_model_node',
            ),
            node(
                predict,
                inputs=[
                    'valid_data',
                    'test_data',
                    'leader_data',
                    'trained_model',
                    'params:data_science',
                ],
                outputs=['model_predictions', 'predict_metrics'],
                name='predict_node',
            ),
        ]
    )
