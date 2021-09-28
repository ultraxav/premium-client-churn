# libs
import datetime
import git
import mlflow
import pandas as pd

from premium_client_churn.__init__ import __version__
from typing import Dict, Any


# nodes
def reporting(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    leader_data: pd.DataFrame,
    model_metrics: Dict[str, Any],
    trained_model: Any,
    model_study: Any,
    model_params: Dict[str, Any],
) -> Any:

    """Returns the summary of the model's performance and tracking

    Args:
        trained_model : Trained model
        model_study   : Model bayesian search study
        model_params  : Summary parameters
        split_params  : Split data parameters
    Returns:
        summary : Model summary

    """
    train_summary = {
        'sample_size': train_data.shape[0],
        'date_from': str(train_data['foto_mes'].min()),
        'date_to': str(train_data['foto_mes'].max()),
        'auc': model_study.best_trial.value,
    }

    valid_summary = {
        'sample_size': valid_data.shape[0],
        'date': str(valid_data['foto_mes'].min()),
        'auc': model_metrics['auc_valid'],
        'gain': model_metrics['gain_valid'],
    }

    test_summary = {
        'sample_size': test_data.shape[0],
        'date': str(valid_data['foto_mes'].min()),
        'auc': model_metrics['auc_test'],
        'gain': model_metrics['gain_test'],
    }

    leaderboard_summary = {
        'sample_size': leader_data.shape[0],
        'date': str(valid_data['foto_mes'].min()),
        'pcutoff': model_metrics['pcutoff'],
    }

    feature_importance = dict(
        zip(
            trained_model.feature_name(),
            trained_model.feature_importance(importance_type='gain'),
        )
    )

    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    best_iteration = trained_model.current_iteration()

    trained_at = str(datetime.datetime.now())

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    try:
        run = mlflow.active_run()
        mlflow_run = run.info.run_id

    except:
        mlflow_run = None

    model_metrics_report = {
        'train': train_summary,
        'valid': valid_summary,
        'test': test_summary,
        'leaderboard': leaderboard_summary,
        'parameters': model_params,
        'best_iteration': best_iteration,
        'feature_importance': feature_importance,
        'trained_at': trained_at,
        'run_id': mlflow_run,
        'commit': sha,
        'version': __version__,
    }

    return model_metrics_report
