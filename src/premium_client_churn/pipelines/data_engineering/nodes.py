# libs
import numpy as np
import pandas as pd

from typing import Any, Dict

# nodes
def clean_data(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Node for cleaning broken features in cretain months and normalizing the target"""

    # Normalizing target class
    data['clase_ternaria'] = (
        data['clase_ternaria'].map({params['target_class']: 1}).fillna(0).astype(int)
    )

    # Categories
    for i in params['bool_to_cat']:
        data[i] = data[i].astype('category')

    # Normalize days to years
    for j in params['day_to_year']:
        data[j] = data[j] / 365

    # Normalize months to years
    data['cliente_antiguedad'] = data['cliente_antiguedad'] / 12

    return data
