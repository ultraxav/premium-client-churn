# libs
import numpy as np
import pandas as pd

from typing import Any, Dict

# nodes
def clean_data(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Node for fixing broken data, transforming time columns into years and normalizing the target
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
    Node for automatic feature engineering
    """
    # Categories
    for i in params['bool_to_cat']:
        data[i] = data[i].astype('category')

    return data
