from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from matplotlib import pyplot as plt

train_features = pd.read_csv('../data/dengue_features_train.csv', index_col=[0, 1, 2])
train_labels = pd.read_csv('../data/dengue_labels_train.csv', index_col=[0, 1, 2])

sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)

sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)
