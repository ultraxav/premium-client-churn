# libs
import lightgbm as lgb
import mlflow
import optuna
import optuna.integration.lightgbm as lgb_optim
import pandas as pd
import warnings

from kedro.framework.session import get_current_session
from typing import Any, Dict

warnings.filterwarnings('ignore', category=UserWarning)

# nodes
def split_data(
    header_data: pd.DataFrame, scoring_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Split the data into train and test datasets

    Args:
        header_data  : Data of given loans
        scoring_data : Processed credit scoring data of loans
        parameters   : Parameter to split train/valid data

    Returns:
        Dict : Train/Test data for model training

    """
    header_data = header_data[['expiry_date', 'score_id', 'delinquency']].copy()

    header_data['expiry_date'] = pd.to_datetime(header_data['expiry_date'])

    header_data['month'] = header_data['expiry_date'].dt.strftime('%Y-%m')

    scoring_data = scoring_data.rename(columns={'id': 'score_id'})

    scoring_data = scoring_data.drop_duplicates(subset='score_id')

    primary_data = header_data.merge(
        scoring_data, on='score_id', suffixes=('_header', '_scoring')
    )

    train = primary_data[primary_data['month'] < parameters['month']]
    test = primary_data[primary_data['month'] >= parameters['month']]

    # Drop lean_created_at y month
    train = train.drop(columns=['expiry_date', 'score_id', 'month'])
    test = test.drop(columns=['expiry_date', 'score_id', 'month'])

    train_data_x = train.drop(columns=['delinquency'])
    train_data_y = train['delinquency']
    test_data_x = test.drop(columns=['delinquency'])
    test_data_y = test['delinquency']

    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    BO_params: Dict[str, Any],
    model_fixed_params: Dict[str, Any],
    model_optim_params: Dict[str, Any],
) -> Any:
    """Trains the model given some data

    Args:
        train_x    : Train Data
        train_y    : Train Labels Data
        test_x     : Train Data
        test_y     : Train Labels Data
        parameters : Dictionary of Parameters
    Returns:
        trained_model : Trained Model
        study         : Train Hyperparamenter Study
        train_params  : Best Model Params

    """
    context = get_current_session().load_context().catalog

    Xt = lgb.Dataset(train_x, label=train_y)

    if BO_params['with_optim'] == True:
        tuner = lgb_optim.LightGBMTunerCV(
            {**model_fixed_params, **{'random_state': BO_params['seed']}},
            Xt,
            nfold=BO_params['nfolds'],
            stratified=False,
            shuffle=False,
            study=optuna.create_study(
                study_name='credit-risk-col-v2', direction='maximize'
            ),
            show_progress_bar=False,
            seed=BO_params['seed'],
            return_cvbooster=True,
            verbose_eval=False,
            optuna_seed=BO_params['seed'],
        )

        tuner.run()

        study = tuner.study
        train_params = tuner.best_params
        train_params.pop('early_stopping_rounds')
        train_params.pop('metric')
        train_params['n_estimators'] = tuner.get_best_booster().current_iteration()[0]

    else:
        model_fixed_params.pop('early_stopping_rounds')
        model_fixed_params.pop('metric')
        model_fixed_params.pop('n_estimators')

        train_params = {
            **model_fixed_params,
            **model_optim_params,
            **{'random_state': BO_params['seed']},
        }

        study = context.load('model_study')

    Xv = lgb.Dataset(test_x, label=test_y)

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
