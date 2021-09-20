# libs
import lightgbm as lgb
import mlflow
import optuna
import optuna.integration.lightgbm as lgb_optim
import pandas as pd

# import warnings

from kedro.framework.session import get_current_session
from typing import Any, Dict

# warnings.filterwarnings('ignore', category=UserWarning)

# nodes
def split_data(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Subsets the dataset to the months needed for training and predicting

    Args:
        data: Feature data
        params: Parameters to split train/valid data

    Returns:
        data: Data to train and predict
        splits: Dictionary of the indices for train, valid, test, and predict
    """
    # train
    train_to = params.experiment_dates.train

    start_from = pd.Series(sorted(data['foto_mes'].unique()))

    start_from = start_from[start_from <= train_to]

    start_from = int(start_from[-params.months_to_train :][:1])

    train_dates = (data['foto_mes'] >= start_from) & (data['foto_mes'] <= train_to)

    # valid
    valid_dates = data['foto_mes'] == params.experiment_dates.valid

    # test
    test_dates = data['foto_mes'] == params.experiment_dates.test

    # predict
    end_in = params.experiment_dates.leader
    predict_dates = data['foto_mes'] == end_in

    # dataset subset
    data = data[(data['foto_mes'] >= start_from) & (data['foto_mes'] <= end_in)]

    splits = {
        'train_dates': train_dates,
        'valid_dates': valid_dates,
        'test_dates': test_dates,
        'predict_dates': predict_dates,
    }

    return data, splits


def train_model(
    data: pd.DataFrame,
    splits: Dict[str, Any],
    params: Dict[str, Any],
) -> Any:
    """
    Trains the model given some data

    Args:
        data: Data to train and predict
        splits: Dictionary of the indices for train, valid, test, and predict
        params: Parameters for model, and bayesian search
    Returns:
        trained_model : Trained Model
        study         : Train Hyperparamenter Study
        train_params  : Best Model Params
    """
    context = get_current_session().load_context().catalog

    Xt = lgb.Dataset(
        data.drop(columns=[params.cols_to_drop])[splits.train_dates],
        label=data[params.target_class][splits.train_dates],
    )

    if params.optim.with_optim == True:
        tuner = lgb_optim.LightGBMTunerCV(
            {**params.model_fixed, **{'random_state': params.optim.seed}},
            Xt,
            stratified=False,
            shuffle=False,
            study=optuna.create_study(
                study_name='premium-clien-churn', direction='maximize'
            ),
            show_progress_bar=False,
            seed=params.optim.seed,
            return_cvbooster=True,
            verbose_eval=False,
            optuna_seed=params.optim.seed,
        )

        tuner.run()

        study = tuner.study
        train_params = tuner.best_params
        train_params.pop('early_stopping_rounds')
        train_params.pop('metric')
        train_params['n_estimators'] = tuner.get_best_booster().current_iteration()[0]

    else:
        params.model_fixed.pop('early_stopping_rounds')
        params.model_fixed.pop('metric')
        params.model_fixed.pop('n_estimators')

        train_params = {
            **params.model_fixed,
            **params.model_optimized,
            **{'random_state': params.optim.seed},
        }

        study = context.load('model_study')

    Xv = lgb.Dataset(
        data.drop(columns=[params.cols_to_drop])[splits.test_dates],
        label=data[params.target_class][splits.test_dates],
    )

    mlflow.lightgbm.autolog(silent=True)

    learner = lgb.train(
        train_params,
        train_set=Xt,
        valid_sets=[Xv],
        verbose_eval=False,
    )

    return learner, study, train_params


def predict(
    data: pd.DataFrame,
    params: Dict[str, Any],
    trained_model: Any,
) -> pd.DataFrame:
    """
    Node for making predictions given a pre-trained model and a dataset.
    Also calculates the optimal cutoff probability to return the final predicitons.

    Args:
        data: Data to make predictions
        params: Month to predict
        trained_model: Trained Model

    Returns:
        model_predictions: Predictions ready to upload to the leaderboard
    """
    data = data.drop(columns='clase_ternaria')
    data = data[data['foto_mes'] == 201904]

    probs = trained_model.predict(data.drop(columns=['numero_de_cliente', 'foto_mes']))

    model_predictions = {
        'Id': data['numero_de_cliente'],
        'Predicted': trained_model.predict(
            data.drop(columns=['numero_de_cliente', 'foto_mes'])
        ),
    }

    return model_predictions
