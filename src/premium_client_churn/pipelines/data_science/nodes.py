# libs
import lightgbm as lgb
import mlflow
import optuna
import optuna.integration.lightgbm as lgb_optim
import pandas as pd
import warnings

from kedro.framework.session import get_current_session
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Any, Dict

warnings.filterwarnings('ignore', category=UserWarning)

# support funcs
def profit_calculator(
    y_true: pd.Series, y_pred: pd.Series, TP_gain: int, FP_gain: int, pcut: float
) -> int:
    """
    Returns the estimated profit of the incetive campaing for the financial institution

    Args:
        y_true: Ture labels
        y_pred: Probability estimate
        TP_gain: Net profit of sending a incentive and keeping a client
        FP_gain: Cost of sending an incentive and losing a client
        pcut: Cutoff probability

    Returns:
        profit: Estimated profit
    """

    y_pred = y_pred.apply(lambda x: int(x > pcut))

    _, fp, _, tp = confusion_matrix(y_true, y_pred).ravel()

    profit = TP_gain * tp + FP_gain * fp

    return profit


# nodes
def split_data(data: pd.DataFrame, params: Dict[str, Any]) -> Any:
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
    train_to = params['experiment_dates']['train']

    start_from = pd.Series(sorted(data['foto_mes'].unique()))

    start_from = start_from[start_from <= train_to]

    start_from = int(start_from[-params['months_to_train'] :][:1])

    train_dates = (data['foto_mes'] >= start_from) & (data['foto_mes'] <= train_to)

    train_data = data[train_dates]

    # valid
    valid_dates = data['foto_mes'] == params['experiment_dates']['valid']

    valid_data = data[valid_dates]

    # test
    test_dates = data['foto_mes'] == params['experiment_dates']['test']

    test_data = data[test_dates]

    # leaderboard
    leader_dates = data['foto_mes'] == params['experiment_dates']['leader']

    leader_data = data[leader_dates]

    return train_data, valid_data, test_data, leader_data


def train_model(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
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
        data.drop(columns=(params['cols_to_drop']))[splits['train_dates']],
        label=data[params['target_class']][splits['train_dates']],
    )

    if params['optim']['with_optim'] == True:
        tuner = lgb_optim.LightGBMTunerCV(
            {**params['model_fixed'], **{'random_state': params['optim']['seed']}},
            Xt,
            stratified=False,
            shuffle=False,
            study=optuna.create_study(
                study_name='premium-client-churn', direction='maximize'
            ),
            show_progress_bar=False,
            seed=params['optim']['seed'],
            return_cvbooster=True,
            verbose_eval=False,
            optuna_seed=params['optim']['seed'],
        )

        tuner.run()

        study = tuner.study
        train_params = tuner.best_params
        train_params.pop('early_stopping_rounds')
        train_params.pop('metric')
        train_params['n_estimators'] = tuner.get_best_booster().current_iteration()[0]

    else:
        params['model_fixed'].pop('early_stopping_rounds')
        params['model_fixed'].pop('metric')
        params['model_fixed'].pop('n_estimators')

        train_params = {
            **params['model_fixed'],
            **params['model_optimized'],
            **{'random_state': params['optim']['seed']},
        }

        study = context.load('model_study')

    Xv = lgb.Dataset(
        data.drop(columns=(params['cols_to_drop']))[splits['test_dates']],
        label=data[params['target_class']][splits['test_dates']],
    )

    mlflow.lightgbm.autolog(silent=True)

    learner = lgb.train(
        train_params,
        train_set=Xt,
        valid_sets=[Xv],
        verbose_eval=False,
    )

    # train_score = study.best_trial.value

    # train_summary = {
    #     'sample_size': train_data.shape[0],
    #     'date_from': str(train_data['expiry_date'].min()),
    #     'date_to': str(train_data['expiry_date'].max()),
    #     'auc': train_score,
    # }

    return learner, study, train_params


def predict(
    data: pd.DataFrame,
    splits: Dict[str, Any],
    trained_model: Any,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Node for making predictions given a pre-trained model and a dataset.
    Also calculates the optimal cutoff probability to return the final predicitons.

    Args:
        data: Data to make predictions
        splits: Dictionary of the indices for train, valid, test, and predict
        params: Month to predict
        trained_model: Trained Model

    Returns:
        model_predictions: Predictions ready to upload to the leaderboard
    """
    X_valid = data.drop(columns=(params['cols_to_drop']))[splits['valid_dates']]
    X_test = data.drop(columns=(params['cols_to_drop']))[splits['test_dates']]
    X_leader = data.drop(columns=(params['cols_to_drop']))[splits['predict_dates']]

    y_valid = data['clase_ternaria'][splits['valid_dates']]
    y_test = data['clase_ternaria'][splits['test_dates']]
    id_leader = data['numero_de_cliente'][splits['predict_dates']]

    preds_valid = trained_model.predict(X_valid)
    preds_test = trained_model.predict(X_test)
    preds_leader = trained_model.predict(X_leader)

    score_valid = roc_auc_score(y_valid, preds_valid)
    score_test = roc_auc_score(y_test, preds_test)

    probs = trained_model.predict(data)

    model_predictions = {
        'Id': id_leader,
        'Predicted': preds_leader,
    }

    model_metrics = {
        'score_valid': score_valid,
        'score_test': score_test,
        'pcutoff': 0,
    }

    return model_predictions, model_metrics
