# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: PremiumClientChurn
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature Selection
#
# We will explore two ways that features can be selected for training a final model, the aim is to select the minimum number of features possible.
#
# This approach has some key qualities:
# * A small number of features will reduce the hardware requirements.
# * ... Also could improve generalization.
# * ... could also induce overfiting, since the selected features are specific to their timeframe, might not be valid in another, for example: seasonality.

# %%
# libs
import matplotlib.pyplot as plt
import pandas as pd
import gc

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

gc.collect()

# %%
data = catalog.load('feature_data')
print(data.shape)
data.head()

# %%
data = data[data['foto_mes'] >= 201906]
data = data[data['foto_mes'] <= 201912]
print(data.shape)
data.head()

# %% [markdown]
# # Recursive Feature Selection
#
# > Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
#
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

# %%
# A simple LightGBM model is created to recursively test the features.
# Data used: from 201901 to 201912 (one year).
# Methodology: 5 fold Cross-validation
learner = LGBMClassifier()

# Continue the search until there is only 1 feature left
min_features_to_select = 1

# Eliminate the worst performing feature on each round
step = 0.05

rfecv = RFECV(
    estimator=learner,
    step=step,
    scoring='roc_auc',
    min_features_to_select=min_features_to_select,
)

# %%
# %%time
rfecv.fit(
    data.drop(columns=['numero_de_cliente', 'foto_mes', 'clase_ternaria']),
    data['clase_ternaria'],
)

# %%
print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('AUC Cross-Validation score')
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()

# %%
data.drop(columns=['numero_de_cliente', 'foto_mes', 'clase_ternaria']).iloc[
    :, rfecv.get_support(indices=True)
].columns

# %%
