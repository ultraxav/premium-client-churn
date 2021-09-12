# libs
from kedro.pipeline import Pipeline, node
from .nodes import clean_data, feat_engineering

# pipelines
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_data,
                inputs=['raw_data', 'params:data_engineering'],
                outputs='primary_data',
                name='clean_data_node',
            ),
            node(
                feat_engineering,
                inputs=['primary_data', 'params:data_engineering'],
                outputs='feature_data',
                name='feat_engineering_node',
            ),
        ]
    )
