from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, GridSearchCV
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

raw_train = pd.read_csv('../data/dengue_features_train.csv')
raw_labels = pd.read_csv('../data/dengue_labels_train.csv')
raw_test = pd.read_csv('../data/dengue_features_test.csv')  ##### Import CSV #####
raw_train2 = pd.read_csv('../data/dengue_features_train.csv')

raw_train = raw_train.drop(['week_start_date'], axis=1)  ##### KOLON İPTALİ #####

raw_train['total_cases'] = raw_labels['total_cases']  ##### TRAIN'E TOTAL_CASE KOLONU EKLE #####
raw_train2['total_cases'] = raw_labels['total_cases']

sj = raw_train[raw_train.city == 'sj'].copy()
sjtest = raw_test[raw_test.city == 'sj'].copy()  ##### SJ IQ AYIRIMI YAP #####
iq = raw_train[raw_train.city == 'iq'].copy()
iqtest = raw_test[raw_test.city == 'iq'].copy()

kolonlar = ['ndvi_ne', 'ndvi_nw',
           'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
           'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
           'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
           'reanalysis_precip_amt_kg_per_m2',
           'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
           'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
           'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
           'station_min_temp_c', 'station_precip_mm']


for v in kolonlar:
    sj.loc[:, v] = [min(x, sjtest[v].max()) for x in sj[v]]  ##### Outlier Çıkarma #####   ### isolotion forest ve LocalOutlierFactor denendi grafik çizildi
    sj.loc[:, v] = [max(x, sjtest[v].min()) for x in sj[v]]  #### Min Max  remover outlier detection


def prepare(frame):
    sicaklik = frame.loc[:, ['station_avg_temp_c',
                             'reanalysis_min_air_temp_k',
                             'station_min_temp_c', 'reanalysis_dew_point_temp_k',
                             'reanalysis_air_temp_k']]  ##### Verileri Oranla #####
    sicaklik_ort = pd.DataFrame(MinMaxScaler().fit_transform(sicaklik), columns=sicaklik.columns)
    frame.loc[:, 'temps_mean'] = sicaklik_ort.mean(axis=1)

    haftasal = [11, 30]  #### birçok kesme noktası denendi en uygunu 11 30 gibi ####
    frame['lagged_winter'] = np.where((frame.weekofyear < haftasal[0]), 1, 0)

    frame['lagged_spring'] = np.where((frame.weekofyear >= haftasal[0]) &
                                      (frame.weekofyear < haftasal[1]), 1, 0)
    frame['lagged_summer'] = np.where((frame.weekofyear >= haftasal[1]), 1, 0)

    gerekenler = ['total_cases', 'lagged_spring', 'lagged_summer', 'lagged_winter',
                  'station_max_temp_c', 'temps_mean', 'reanalysis_relative_humidity_percent',
                  'reanalysis_specific_humidity_g_per_kg']  ##### Gereksiz kolonları çıkar #####
    for col in frame.columns:
        if col not in gerekenler:
            frame = frame.drop(col, axis=1)

    kaydir = ['station_max_temp_c', 'temps_mean', 'reanalysis_relative_humidity_percent',
                'reanalysis_specific_humidity_g_per_kg']
    for i in kaydir:
        frame[i + '_1lag'] = frame[i].shift(-1)
        frame[i + '_2lag'] = frame[i].shift(-2)  ##### 4 tane gecikme kolonu ekle ######
        frame[i + '_3lag'] = frame[i].shift(-3)
        frame[i + '_4lag'] = frame[i].shift(-4)

    frame = frame.fillna(method='ffill')  #### boşluk hala varsa ###
    return frame

sj = prepare(sj)

sj.to_csv("../data/sjtest_ciktisi.csv")

raw_train = raw_train.drop(['year'], axis=1)
raw_train = raw_train.drop(['weekofyear'], axis=1)
sj_test = raw_test[raw_test.city == 'sj'].copy()
sj_test = prepare(sj_test)

sj_X = sj.drop(['total_cases'], axis=1)
sj_y = sj.total_cases

X_train_sj, X_test_sj, y_train_sj, y_test_sj = train_test_split(
    sj_X, sj_y, test_size=0.2, shuffle=False)

# param_grid = {
#     'max_depth': [5, 10, 20, 35, 50,100],
#     'max_features': [2, 5],
#     'min_samples_leaf': [2, 3, 4],  ##### Random Grid Search için gerekli ####
#     'min_samples_split': [2, 3, 4],
#     'n_estimators': [100, 200, 400, 500],
# }


sj_rf_params = {'max_depth': 35,
                'bootstrap': True,
                'max_features': 5,
                'min_samples_leaf': 6,
                'min_samples_split': 5,
                'n_estimators': 400}


sj_rfr = RandomForestRegressor(**sj_rf_params, criterion='mae')

# clf = RandomizedSearchCV(sj_rfr, param_grid, n_iter=1000, cv=5, verbose=0, n_jobs=-1)
# search = clf.fit(sj_X, sj_X)

sj_rfr.fit(sj_X, sj_y)
sj_pred = sj_rfr.predict(sj_test).astype(int)


for v in kolonlar:
    iq.loc[:, v] = [min(x, iqtest[v].max()) for x in iq[v]]
    iq.loc[:, v] = [max(x, iqtest[v].min()) for x in iq[v]]

sq_regressor3 = RandomForestRegressor(**sj_rf_params, criterion='mae')   ### test metodu mean absolute error
sq_regressor3.fit(X_train_sj, y_train_sj)
tahmin = sq_regressor3.predict(X_test_sj).astype(int)
print("sj cross validation MAE:", mean_absolute_error(tahmin, y_test_sj))

iq = raw_train2[raw_train2.city == 'iq'].copy()


def process_iq(inputdata):
    feats = inputdata.fillna(method='ffill')
    mevsimler = [14, 28, 37]  ## bir çok deneme sonucu
    feats['lagged_guz'] = np.where((feats.weekofyear < mevsimler[0]), 1, 0)

    feats['lagged_kis'] = np.where((feats.weekofyear >= mevsimler[0]) &
                                   (feats.weekofyear < mevsimler[1]), 1, 0)
    feats['lagged_bahar'] = np.where((feats.weekofyear >= mevsimler[1]) &
                                     (feats.weekofyear < mevsimler[2]), 1, 0)
    feats['lagged_yaz'] = np.where((feats.weekofyear >= mevsimler[2]), 1, 0)

    columns = ['total_cases','reanalysis_specific_humidity_g_per_kg',
            'lagged_bahar', 'lagged_yaz', 'lagged_guz', 'lagged_kis','precipitation_amt_mm', 'station_avg_temp_c',
            'reanalysis_min_air_temp_k', 'reanalysis_dew_point_temp_k', 'station_min_temp_c'
            ]

    for col in feats.columns:
        if col not in columns:
            feats = feats.drop(col, axis=1)

    to_shift = ['station_avg_temp_c', 'reanalysis_min_air_temp_k', 'station_min_temp_c',
                'reanalysis_dew_point_temp_k',
                'reanalysis_specific_humidity_g_per_kg', 'precipitation_amt_mm']

    for i in to_shift:
        feats[i + '_1lag'] = feats[i].shift(-1)
        feats[i + '_2lag'] = feats[i].shift(-2)
        feats[i + '_3lag'] = feats[i].shift(-3)
        feats[i + '_4lag'] = feats[i].shift(-4)

    feats = feats.fillna(method='ffill')
    return feats


iq = process_iq(iq)
iq_X = iq.drop(['total_cases'], axis=1)
iq_y = iq.total_cases

X_train_iq, X_test_iq, y_train_iq, y_test_iq = train_test_split(    #### train test split #### kendi testimiz için gerekli
    iq_X, iq_y, test_size=0.2, shuffle=False)

iq_rf_params = {'max_depth': 35,
                'max_features': 5,
                'min_samples_leaf': 3,    ##### hyperparameter dene ####
                'min_samples_split': 2,
                'n_estimators': 2000}

iq_regressor = RandomForestRegressor(**iq_rf_params, criterion='mae')
iq_regressor.fit(iq_X, iq_y)
print('Bitti!')

iq_regressor_test = RandomForestRegressor(**iq_rf_params, criterion='mae')
iq_regressor_test.fit(X_train_iq, y_train_iq)                                             #### train test split #### kendi testimiz için gerekli
tahmin = iq_regressor_test.predict(X_test_iq).astype(int)
print("iq cross validation MAE:", mean_absolute_error(tahmin, y_test_iq))

iq_test = raw_test[raw_test.city == 'iq'].copy()
iq_test = process_iq(iq_test)

iq_pred = iq_regressor.predict(iq_test).astype(int)

feat_importances = pd.Series(sj_rfr.feature_importances_, index=sj_X.columns)
feat_importances.nlargest(35).plot(kind='barh')                                          ##### importance grafiği ######
plt.show()

submission = pd.read_csv('../data/submission_format.csv', index_col=[0, 1, 2])
submission.total_cases = np.concatenate([sj_pred, iq_pred])
submission.to_csv("../data/final.csv")
