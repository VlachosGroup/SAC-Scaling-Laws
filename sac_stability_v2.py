# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:48:34 2019

@author: wangyf
"""

'''
Test SAC_stability V2
'''
import os
import regression_tools as rtools

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, cross_val_score, train_test_split

import pandas as pd
import numpy as np
import pickle 

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import matplotlib


from sklearn.linear_model import LassoCV, lasso_path, Lasso



font = {'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


'''
read adsoprtion energy and barder charge from a csv file
'''
data = pd.read_csv('pca_data.csv', header = 0)

metal = np.array(data['Metal'])
Ebulk = np.array(data['Ebulk'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])


X_init = np.stack((Ebulk, Ebind, Ebind/Ebulk, Ebulk/Ebind)  ,1) 
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_init)

#%%

pi = poly.powers_
pi_unrepeated =  [0, 1, 2, 3, 4, 5, 8, 9, 10, 12,  14]
X_poly_unrepeated = X_poly[:, np.array(pi_unrepeated)]
X = X_poly_unrepeated
y = Ea
fit_int_flag = False # Not fitting for intercept, as the first coefficient is the intercept

#%% Preparation before regression
# Train test split, save 10% of data point to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
                       
# The alpha grid used for plotting path
alphas_grid = np.logspace(0, -3, 20)

# Cross-validation scheme                                  
rkf = RepeatedKFold(n_splits = 10, n_repeats = 1 , random_state = 0)


# Explicitly take out the train/test set
X_cv_train, y_cv_train, X_cv_test, y_cv_test = [],[],[],[]
for train_index, test_index in rkf.split(X):
    X_cv_train.append(X[train_index])
    y_cv_train.append(y[train_index])
    X_cv_test.append(X[test_index])
    y_cv_test.append(y[test_index])

#%% LASSO regression
'''   
# LassoCV to obtain the best alpha, the proper training of Lasso
'''
model_name = 'lasso'
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

lasso_cv  = LassoCV(cv = rkf,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state=0)
lasso_cv.fit(X_train, y_train)

# the optimal alpha from lassocv
lasso_alpha = lasso_cv.alpha_
# Coefficients for each term
lasso_coefs = lasso_cv.coef_
# The original intercepts 
lasso_intercept = lasso_cv.intercept_

# Access the errors 
y_predict_test = lasso_cv.predict(X_test)
y_predict_train = lasso_cv.predict(X_train)


lasso_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
lasso_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))

##Use alpha grid prepare for lassopath
lasso_RMSE_path, lasso_coef_path = rtools.cal_path(alphas_grid, Lasso, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
##lasso_path to get alphas and coef_path, somehow individual CV does not work
#lasso_alphas, lasso_coef_path, _ = lasso_path(X_train, y_train, alphas = alphas_grid, fit_intercept=fit_int_flag)
rtools.plot_path(X, y, lasso_alpha, alphas_grid, lasso_RMSE_path, lasso_coef_path, lasso_cv, model_name, output_dir)
lasso_RMSE, lasso_r2 = rtools.cal_performance(X, y, lasso_cv, model_name)

# The indices for non-zero coefficients/significant cluster interactions 
J_index = np.nonzero(lasso_coefs)[0]
# The number of non-zero coefficients/significant cluster interactions  
n_nonzero = len(J_index)
# The values of non-zero coefficients/significant cluster interactions  
J_nonzero = lasso_coefs[J_index] 