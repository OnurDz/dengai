#Linear, Lasso, Ridge, Support Vector Regression models for DengAI
#You can run the code and submit the output to the DengAI competition

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from matplotlib import pyplot as plt

train_features = pd.read_csv('../data/dengue_features_train.csv', index_col=[0, 1, 2])
train_labels = pd.read_csv('../data/dengue_labels_train.csv', index_col=[0, 1, 2])

test_features = pd.read_csv('../data/dengue_features_test.csv', index_col=[0, 1, 2])

submission_format = pd.read_csv('../data/submission_format.csv')

sj_train_features = train_features.loc['sj'].copy()
sj_train_labels = train_labels.loc['sj'].copy()
iq_train_features = train_features.loc['iq'].copy()
iq_train_labels = train_labels.loc['iq'].copy()

sj_test_features = test_features.loc['sj'].copy()
iq_test_features = test_features.loc['iq'].copy()

sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)

sj_test_features.drop('week_start_date', axis=1, inplace=True)
iq_test_features.drop('week_start_date', axis=1, inplace=True)

sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

sj_test_features.fillna(method='ffill', inplace=True)
iq_test_features.fillna(method='ffill', inplace=True)

sj_X_train, sj_X_test, sj_y_train, sj_y_test = train_test_split(sj_train_features,sj_train_labels,test_size=0.4)
iq_X_train, iq_X_test, iq_y_train, iq_y_test = train_test_split(iq_train_features,iq_train_labels,test_size=0.4)

def main():
    y_pred=support_vector_regressor()
    submission_format['total_cases']=y_pred
    print('---------------')
    submission_format.to_csv ('../data/submission_format.csv', index=False)
    # linear_regression()
    # ridge_regression()
    # lasso_regression()
    # support_vector_regressor()


def support_vector_regressor():
    sj_sc_X = StandardScaler()
    sj_sc_y = StandardScaler()
    sj_X = sj_sc_X.fit_transform(sj_train_features)
    sj_y = sj_sc_y.fit_transform(sj_train_labels)

    iq_sc_X = StandardScaler()
    iq_sc_y = StandardScaler()
    iq_X = iq_sc_X.fit_transform(iq_X_train)
    iq_y = iq_sc_y.fit_transform(iq_y_train)
    C=[0.001,0.01,0.1,1,10,100]
    gamma=[0.001,0.01,0.1,1,10]
    for c in C:
        for g in gamma:
            sj_svr = SVR(kernel='rbf',C=1,gamma=0.01) 
            iq_svr = SVR(kernel='rbf',C=1,gamma=0.01)

            # sj_svr_lin = SVR(kernel='linear',C=c,gamma=g)
            # sj_svr_poly = SVR(kernel='poly',C=c,gamma=g)
            # iq_svr_lin = SVR(kernel='linear',C=c,gamma=g)
            # iq_svr_poly = SVR(kernel='poly',C=c,gamma=g)

            sj_svr.fit(sj_X,sj_y)
            iq_svr.fit(iq_X,iq_y)

            # sj_svr_lin.fit(sj_X,sj_y)
            # iq_svr_lin.fit(iq_X,iq_y)

            # sj_svr_poly.fit(sj_X,sj_y)
            # iq_svr_poly.fit(iq_X,iq_y)

            sj_y_pred = sj_sc_y.inverse_transform ((sj_svr.predict (sj_sc_X.transform(sj_test_features))))
            iq_y_pred = iq_sc_y.inverse_transform ((iq_svr.predict (iq_sc_X.transform(iq_test_features))))

            # sj_lin_y_pred = sj_sc_y.inverse_transform ((sj_svr_lin.predict (sj_sc_X.transform(sj_X_test))))
            # iq_lin_y_pred = iq_sc_y.inverse_transform ((iq_svr_lin.predict (iq_sc_X.transform(iq_X_test))))
            
            # sj_poly_y_pred = sj_sc_y.inverse_transform ((sj_svr_poly.predict (sj_sc_X.transform(sj_X_test))))
            # iq_poly_y_pred = iq_sc_y.inverse_transform ((iq_svr_poly.predict (iq_sc_X.transform(iq_X_test))))

            y_pred = np.concatenate((sj_y_pred, iq_y_pred), axis=None) 
            
            # y_pred_lin = np.concatenate((sj_lin_y_pred, iq_lin_y_pred), axis=None) 
            # y_pred_poly = np.concatenate((sj_poly_y_pred, iq_poly_y_pred), axis=None) 
            # y_test = np.concatenate((sj_y_test, iq_y_test), axis=None)

            int_y_pred=np.rint(y_pred).astype(int)
            # int_y_pred_lin=np.rint(y_pred_lin)
            # int_y_pred_poly=np.rint(y_pred_poly)

            # print('C=', c, ' gamma:',g)
            # print("Support Vector Regression_RBF Kernel")
            # print("Mean absolute error: ", mean_absolute_error(int_y_pred, y_test))
            # print("Support Vector Regression_Linear Kernel")
            # print("Mean absolute error: ", mean_absolute_error(int_y_pred_lin, y_test))
            # print("Support Vector Regression_Poly Kernel")
            # print("Mean absolute error: ", mean_absolute_error(int_y_pred_poly, y_test))
            # print('-------------------------------------------------------')

    return int_y_pred

def linear_regression():
    sj_lr = LinearRegression(normalize=True)
    sj_lr.fit(sj_X_train,sj_y_train)
    sj_y_pred = sj_lr.predict(sj_test_features)

    iq_lr = LinearRegression()
    iq_lr.fit(iq_X_train,iq_y_train)
    iq_y_pred = iq_lr.predict(iq_test_features)
    # iq_y_pred = iq_lr.predict(iq_X_test)
    y_pred = np.concatenate((sj_y_pred, iq_y_pred), axis=None)

    # y_test = np.concatenate((sj_y_test, iq_y_test), axis=None)
    int_y_pred=np.rint(y_pred).astype(int)
    # coef=np.sum(sj_lr.coef_!=0)
    # print('-------------------------------------------------------')
    # print('Linear')
    # print("Mean absolute error: ", mean_absolute_error(int_y_pred,y_test))
    # print('Kullanılan öznitelik sayısı: ', coef)
    # print('-------------------------------------------------------')
    return int_y_pred

def ridge_regression():
    print('Ridge')
    alfa = [0,0.001,0.001,0.01,0.1,1]
    # mae=[]
    for updated_alfa in alfa:
        sj_r = Ridge(alpha=updated_alfa,normalize=True)
        sj_r.fit(sj_X_train,sj_y_train)
        sj_y_pred = sj_r.predict(sj_test_features)

        iq_r = Ridge(alpha=updated_alfa,normalize=True)
        iq_r.fit(iq_X_train,iq_y_train)
        iq_y_pred = iq_r.predict(iq_X_test)

        # y_test = np.concatenate((sj_y_test, iq_y_test), axis=None)
        y_pred = np.concatenate((sj_y_pred, iq_y_pred), axis=None)
        int_y_pred=np.rint(y_pred).astype(int)

        # coef=np.sum(sj_r.coef_!=0)
        # error=mean_absolute_error(int_y_pred, sj_y_test)
        # mae.append(error)
        # print('Alfa: ' + str(updated_alfa))
        # print("Mean absolute error: ", error )
        # print('Selected feature numbers: ',coef)
    
    # plt.figure(figsize=(12,6))
    # plt.plot(alfa,mae)
    # plt.xlabel('Alfa Value')
    # plt.ylabel('Mean Absolute Error')
    # plt.title("Ridge Regression MAE According to Alfa")
    # plt.xscale("log")
    # plt.show()
    # print('-------------------------------------------------------')
    return int_y_pred
        

def lasso_regression():    
    print('Lasso')
    alfa = [0.001,0.01,0.1]
    # mae=[]
    for updated_alfa in alfa:
        sj_lasso = Lasso(alpha=updated_alfa,max_iter=10e5,normalize=True)
        sj_lasso.fit(sj_X_train,sj_y_train)
        sj_y_pred=sj_lasso.predict(sj_X_test)

        iq_lasso = Lasso(alpha=updated_alfa,max_iter=10e5,normalize=True)
        iq_lasso.fit(iq_X_train,iq_y_train)
        iq_y_pred = iq_lasso.predict(iq_X_test)

        # y_test = np.concatenate((sj_y_test, iq_y_test), axis=None)
        y_pred = np.concatenate((sj_y_pred, iq_y_pred), axis=None)
        int_y_pred=np.rint(y_pred).astype(int)
    #     coef=max(np.sum(sj_lasso.coef_!=0),np.sum(iq_lasso.coef_!=0))
    #     error=mean_absolute_error(int_y_pred, sj_y_test)
    #     mae.append(error)
    #     print('Alfa: ' + str(updated_alfa))
    #     print("Mean absolute error: ", error )
    #     print('Selected feature numbers: ',coef)

    # print('-------------------------------------------------------')
    return int_y_pred


if __name__ == "__main__":
    main()