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
                outputs=['train_data', 'splits'],
                name='split_data_node',
            ),
            node(
                train_model,
                inputs=['feature_data', 'params:experiment_dates'],
                outputs=['trained_model', 'walk_gains', 'model_params'],
                name='train_model_node',
            ),
            node(
                predict,
                inputs=['feature_data', 'params:experiment_dates', 'trained_model'],
                outputs='predictions',
                name='predict_node',
            ),
        ]
    )
