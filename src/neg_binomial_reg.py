from __future__ import print_function
from __future__ import division

import pandas as pd
from patsy import dmatrices
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

train_features = pd.read_csv('../data/dengue_features_train.csv', header=0, infer_datetime_format=True, index_col=[0, 1, 2])
train_labels = pd.read_csv('../data/dengue_labels_train.csv', index_col=[0, 1, 2])

test_features = pd.read_csv('../data/dengue_features_test.csv', header=0, infer_datetime_format=True, index_col=[0, 1, 2])

sj_train_features = train_features.loc['sj'].copy()
sj_train_labels = train_labels.loc['sj'].copy()
sj_y = sj_train_labels['total_cases']

sj_test_features = test_features.loc['sj'].copy()


iq_train_features = train_features.loc['iq'].copy()
iq_train_labels = train_labels.loc['iq'].copy()

sj_train_features.drop('week_start_date', axis=1, inplace=True)
sj_test_features.drop('week_start_date', axis=1, inplace=True)

iq_train_features.drop('week_start_date', axis=1, inplace=True)

sj_train_features.fillna(method='ffill', inplace=True)
sj_test_features.fillna(method='ffill', inplace=True)

iq_train_features.fillna(method='ffill', inplace=True)

sj_train_features['total_case'] = sj_y
sj_test_features['total_case'] = 1

expr = """total_case ~ ndvi_ne + ndvi_nw + ndvi_se + ndvi_sw + precipitation_amt_mm + reanalysis_air_temp_k + reanalysis_avg_temp_k + reanalysis_dew_point_temp_k
       + reanalysis_max_air_temp_k + reanalysis_min_air_temp_k + reanalysis_precip_amt_kg_per_m2 + reanalysis_relative_humidity_percent + reanalysis_sat_precip_amt_mm 
       + reanalysis_specific_humidity_g_per_kg + reanalysis_tdtr_k + station_avg_temp_c + station_diur_temp_rng_c + station_max_temp_c + station_min_temp_c + station_precip_mm  """

y_train, x_train = dmatrices(expr, sj_train_features, return_type='dataframe')
y_test, x_test = dmatrices(expr, sj_test_features, return_type='dataframe')

poisson_training_results = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
#print(poisson_training_results.summary())
#print(poisson_training_results.mu)
#print(len(poisson_training_results.mu))

sj_train_features['TC_LAMBDA'] = poisson_training_results.mu

sj_train_features['AUX_OLS_DEP'] = sj_train_features.apply(lambda x: ((x['total_case'] - x['TC_LAMBDA'])**2 - x['total_case']) / x['TC_LAMBDA'], axis=1)
ols_expr = """AUX_OLS_DEP ~ TC_LAMBDA - 1"""
aux_olsr_results = smf.ols(ols_expr, sj_train_features).fit()
# print(aux_olsr_results.params)
# print(aux_olsr_results.tvalues)

nb2_training_results = sm.GLM(y_train, x_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
#print(nb2_training_results.summary())
nb2_predictions = nb2_training_results.get_prediction(x_test)

predictions_summary_frame = nb2_predictions.summary_frame()
print(predictions_summary_frame)