# libs
import numpy as np
import pandas as pd

from typing import Any, Dict

# nodes
def clean_data(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Node for fixing broken data, transforming time columns into years, and normalizing the target.

    Args:
        data: Raw data
        params: Params to normalize target class

    Returns:
        data: Clean data
    """
    # Normalizing target class
    data['clase_ternaria'] = (
        data['clase_ternaria'].map({params['target_class']: 1}).fillna(0).astype(int)
    )

    # Broken data
    data.loc[data['foto_mes'] <= 201905, 'ctransferencias_recibidas'] = np.nan
    data.loc[data['foto_mes'] <= 201905, 'mtransferencias_recibidas'] = np.nan

    # Year 2017
    data.loc[data['foto_mes'] == 201701, 'ccajas_consultas'] = np.nan
    data.loc[data['foto_mes'] == 201702, 'ccajas_consultas'] = np.nan

    # Year 2018
    data.loc[data['foto_mes'] == 201801, 'ccajas_consultas'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'ccajas_depositos'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'ccajas_extracciones'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'ccajas_otras'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'ccajas_transacciones'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'ccallcenter_transacciones'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'chomebanking_transacciones'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'cprestamos_personales'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'internet'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'mprestamos_hipotecarios'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'mprestamos_personales'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'tcallcenter'] = np.nan
    data.loc[data['foto_mes'] == 201801, 'thomebanking'] = np.nan
    data.loc[data['foto_mes'] == 201806, 'ccallcenter_transacciones'] = np.nan
    data.loc[data['foto_mes'] == 201806, 'tcallcenter'] = np.nan

    # Year 2019
    data.loc[data['foto_mes'] == 201904, 'ctarjeta_visa_debitos_automaticos'] = np.nan
    data.loc[data['foto_mes'] == 201904, 'mttarjeta_visa_debitos_automaticos'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'ccomisiones_otras'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mactivos_margen'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mcomisiones'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mcomisiones_otras'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mpasivos_margen'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mrentabilidad'] = np.nan
    data.loc[data['foto_mes'] == 201905, 'mrentabilidad_annual'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'ccajeros_propios_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'ccomisiones_otras'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'chomebanking_transacciones'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'ctarjeta_master_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'ctarjeta_visa_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mactivos_margen'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mcajeros_propios_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mcomisiones'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mcomisiones_otras'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mpasivos_margen'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mrentabilidad'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mrentabilidad_annual'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mtarjeta_master_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 201910, 'mtarjeta_visa_descuentos'] = np.nan

    # Year 2020
    data.loc[data['foto_mes'] == 202001, 'cliente_vip'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'ccajeros_propios_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'cliente_vip'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'ctarjeta_master_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'ctarjeta_visa_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'mcajeros_propios_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'mtarjeta_master_descuentos'] = np.nan
    data.loc[data['foto_mes'] == 202002, 'mtarjeta_visa_descuentos'] = np.nan

    # Normalize days to years
    for i in params['day_to_year']:
        data[i] = data[i] / 365

    # Normalize months to years
    data['cliente_antiguedad'] = data['cliente_antiguedad'] / 12

    return data


def feat_engineering(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Node for feature engineering.

    Args:
        data: Clean data
        params: Params to make feature engineering

    Returns:
        data: Feature data
    """
    # Calculates de median of mpayroll for every month
    sueldos = data[['foto_mes', 'mpayroll']]

    sueldos = sueldos[sueldos['mpayroll'] > 0].reset_index(drop=True)
    sueldos['foto_mes'] = sueldos.loc[:, 'foto_mes'].astype('str')

    sueldos_ireg = sueldos[
        (sueldos['foto_mes'].str.contains('06'))
        | (sueldos['foto_mes'].str.contains('12'))
    ]
    sueldos_ireg['foto_mes'] = sueldos_ireg.loc[:, 'foto_mes'].astype('int').copy()

    sueldos_reg = sueldos[
        ~(
            (sueldos['foto_mes'].str.contains('06'))
            | (sueldos['foto_mes'].str.contains('12'))
        )
    ]
    sueldos_reg['foto_mes'] = sueldos_reg.loc[:, 'foto_mes'].astype('int').copy()

    agg_funcs = {'month_median': ('mpayroll', 'median')}

    normalizers_ireg = sueldos_ireg.groupby('foto_mes').agg(**agg_funcs)
    normalizers_ireg['month_median'] = normalizers_ireg['month_median'].shift(1)

    normalizers_reg = sueldos_reg.groupby('foto_mes').agg(**agg_funcs)
    normalizers_reg['month_median'] = normalizers_reg['month_median'].shift(1)

    normalizers = pd.concat([normalizers_reg, normalizers_ireg]).sort_index()

    median_vector = data[['foto_mes']].merge(normalizers, on='foto_mes', how='left')

    median_vector = median_vector['month_median']

    # Applies the median vector to all pesos columns to eliminate the effect of inflation
    for i in params['cols_pesos']:
        data[i] = data[i] / median_vector

    # Applies lags to historic columns
    data = data.sort_values(by=['numero_de_cliente', 'foto_mes']).reset_index(drop=True)

    lag_data = data.groupby('numero_de_cliente').shift(params['lag_qty'])

    lag_data = lag_data[params['cols_to_lag']]

    lag_data = lag_data.add_prefix('lag_')

    data = pd.concat([data, lag_data], axis=1)

    return data
