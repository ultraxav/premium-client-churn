# libs
from kedro.pipeline import Pipeline, node
from .nodes import clean_data

# pipelines
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_data,
                inputs=['raw_data', 'params:clean_data'],
                outputs='intermediate_data',
                name='clean_data_node',
            )
        ]
    )
