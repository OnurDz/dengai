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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder, RobustScaler, KernelCenterer, \
    StandardScaler

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





kolonlar = [
            'ndvi_ne', 'ndvi_nw',
            'ndvi_se', 'ndvi_sw',
            'precipitation_amt_mm',
            'reanalysis_air_temp_k',
            'reanalysis_avg_temp_k',
            'reanalysis_dew_point_temp_k',
            'reanalysis_max_air_temp_k',
            'reanalysis_min_air_temp_k',
            'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_relative_humidity_percent',
            'reanalysis_sat_precip_amt_mm',
            'reanalysis_specific_humidity_g_per_kg',
            'reanalysis_tdtr_k',
            'station_avg_temp_c',
            'station_diur_temp_rng_c',
            'station_max_temp_c',
            'station_min_temp_c',
            'station_precip_mm'
            ]



for kolon in kolonlar:
    sj.loc[:, kolon] = [max(sjtest[kolon].min(), sample) for sample in sj[kolon]]
    sj.loc[:, kolon] = [min(sjtest[kolon].max(), sample) for sample in sj[kolon]]    ##### Outlier Çıkarma #####   ### isolotion forest ve LocalOutlierFactor denendi grafik çizildi


def prepare(sj):
    sicaklik = sj.loc[:, ['station_avg_temp_c',
                             'reanalysis_min_air_temp_k',
                             'station_min_temp_c', 'reanalysis_dew_point_temp_k',
                             'reanalysis_air_temp_k']]  ##### Verileri Oranla #####
    sicaklik_ort = pd.DataFrame(StandardScaler().fit_transform(sicaklik), columns=sicaklik.columns)
    sj.loc[:, 'sicaklik_ort'] = sicaklik_ort.mean(axis=1)


    #MaxAbsScaler
    #MinMaxScaler
    #OneHotEncoder ## hata verdi        denenenler isolation forest ve LOF metotları da denendi
    #RobustScaler
    #KernelCenterer  ##hata
    #StandardScaler



    haftasal = [11, 30]  #### birçok kesme noktası denendi en uygunu 11 30 gibi ####
    sj['mevsim_spring'] = np.where((sj.weekofyear >= haftasal[0]) & (sj.weekofyear < haftasal[1]), 1, 0)
    sj['mevsim_winter'] = np.where((sj.weekofyear < haftasal[0]), 1, 0)
    sj['mevsim_summer'] = np.where((sj.weekofyear >= haftasal[1]), 1, 0)

    gerekenler = ['total_cases', 'mevsim_spring', 'mevsim_summer', 'mevsim_winter', 'station_max_temp_c', 'sicaklik_ort', 'reanalysis_relative_humidity_percent',
                  'reanalysis_specific_humidity_g_per_kg']  ##### Gereksiz kolonları çıkar #####
    for col in sj.columns:
        if col not in gerekenler:
            sj = sj.drop(col, axis=1)

    kaydir = ['station_max_temp_c', 'sicaklik_ort', 'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg']

    for i in kaydir:
        sj[i + '_1week'] = sj[i].shift(-1)
        sj[i + '_2week'] = sj[i].shift(-2)  ##### 4 tane gecikme kolonu ekle , 1 , 2, 3 ,4 ayrı ayrı denendi ######
        sj[i + '_3week'] = sj[i].shift(-3)

    sj = sj.fillna(method='ffill')  #### boşluk hala varsa ###
    return sj


# model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=1.0)          ### isolationForest


sj = prepare(sj)
def process_iq(inputdata):
    features = inputdata.fillna(method='ffill')
    mevsimler = [14, 28, 37]  ## bir çok deneme sonucu


    features['mevsim_kis'] = np.where((features.weekofyear >= mevsimler[0]) & (features.weekofyear < mevsimler[1]), 1,
                                      0)
    features['mevsim_yaz'] = np.where((features.weekofyear >= mevsimler[2]), 1, 0)
    features['mevsim_bahar'] = np.where((features.weekofyear >= mevsimler[1]) &
                                        (features.weekofyear < mevsimler[2]), 1, 0)
    features['mevsim_guz'] = np.where((features.weekofyear < mevsimler[0]), 1, 0)

    columns = ['total_cases', 'reanalysis_specific_humidity_g_per_kg', 'mevsim_bahar', 'mevsim_yaz', 'mevsim_guz',
               'mevsim_kis', 'precipitation_amt_mm', 'station_avg_temp_c',
               'reanalysis_min_air_temp_k', 'reanalysis_dew_point_temp_k', 'station_min_temp_c'
               ]

    for col in features.columns:
        if col not in columns:
            features = features.drop(col, axis=1)

    kaydir_iq = ['reanalysis_min_air_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_min_temp_c',
                 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'precipitation_amt_mm']

    for i in kaydir_iq:
        features[i + '_1week'] = features[i].shift(-1)
        features[i + '_2week'] = features[i].shift(-2)
        features[i + '_3week'] = features[i].shift(-3)

    features = features.fillna(method='ffill')  ##boşluk kalmadığına emin ol forward filling
    return features

sj.to_csv("../data/sjtest_ciktisi.csv")

raw_train = raw_train.drop(['year'], axis=1)
raw_train = raw_train.drop(['weekofyear'], axis=1)
sj_test = raw_test[raw_test.city == 'sj'].copy()
sj_test = prepare(sj_test)

sj_X = sj.drop(['total_cases'], axis=1)
sj_y = sj.total_cases

sj_train_X, sj_test_X, sj_trainY, sjtestY = train_test_split(
    sj_X, sj_y, test_size=0.2, shuffle=False)

# param_grid = {
#     'max_depth': [5, 10, 20, 35, 50,100],
#     'max_features': [2, 5],
#     'min_samples_leaf': [2, 3, 4],  ##### Random Grid Search için gerekli ####
#     'min_samples_split': [2, 3, 4],
#     'n_estimators': [100, 200, 400, 500],
# }
# model = RandomForestRegressor()
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid.fit(sj_X, sj_y)

sj_regressor_params = {'max_depth': 35,
                'bootstrap': True,
                'max_features': 5,
                'min_samples_leaf': 6,
                'min_samples_split': 5,
                'n_estimators': 400}

sj_regressor = RandomForestRegressor(**sj_regressor_params, criterion='mae')

# clf = RandomizedSearchCV(sj_regressor, param_grid, n_iter=1000, cv=5, verbose=0, n_jobs=-1)
# search = clf.fit(sj_X, sj_X)

sj_regressor.fit(sj_X, sj_y)
sj_pred = sj_regressor.predict(sj_test).astype(int)

for duzeltilecek in kolonlar:
    iq.loc[:, duzeltilecek] = [min(iqtest[duzeltilecek].max(), sample) for sample in iq[duzeltilecek]]
    iq.loc[:, duzeltilecek] = [max(iqtest[duzeltilecek].min(), sample) for sample in iq[duzeltilecek]]

sq_regressor3 = RandomForestRegressor(**sj_regressor_params, criterion='mae')  ### test metodu mean absolute error
sq_regressor3.fit(sj_train_X, sj_trainY)
tahmin = sq_regressor3.predict(sj_test_X).astype(int)
print("sj cross validation MAE:", mean_absolute_error(tahmin, sjtestY))

iq = raw_train2[raw_train2.city == 'iq'].copy()


iq = process_iq(iq)
iq_X = iq.drop(['total_cases'], axis=1)
iq_y = iq.total_cases

iq_split_train_X, iq_split_test_X, iq_train_y, iq_text_y = train_test_split(  #### train test split #### kendi testimiz için gerekli
    iq_X, iq_y, test_size=0.2, shuffle=False)


iq_rf_params = {
                'max_depth': 35,
                'max_features': 5,
                'min_samples_leaf': 3,  ##### hyperparameter farklı deenenecek ####
                'min_samples_split': 2,
                'n_estimators': 2000
                }



iq_regressor = RandomForestRegressor(**iq_rf_params, criterion='mae')
iq_regressor.fit(iq_X, iq_y)


iq_regressor_test = RandomForestRegressor(**iq_rf_params, criterion='mae')
iq_regressor_test.fit(iq_split_train_X, iq_train_y)  #### train test split #### kendi testimiz için gerekli
tahmin = iq_regressor_test.predict(iq_split_test_X).astype(int)
print("iq cross validation MAE:", mean_absolute_error(tahmin, iq_text_y))



iq_test = raw_test[raw_test.city == 'iq'].copy()
iq_test = process_iq(iq_test)
iq_pred = iq_regressor.predict(iq_test).astype(int)

submission = pd.read_csv('../data/submission_format.csv', index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_pred, iq_pred])

submission.to_csv("../data/final.csv")


feat_importances = pd.Series(sj_regressor.feature_importances_, index=sj_X.columns)
feat_importances.nlargest(35).plot(kind='barh')  ##### importance grafiği ######
plt.show()


