# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:53:18 2019

@author: yifan
"""

'''
Test SAC_stability
Standize 
Introducing more features
Use experimental values for Ec cohesive
'''

import os
import pickle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV,
                                  Ridge, RidgeCV, enet_path, lasso_path)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (LeaveOneOut, RepeatedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import regression_tools as rtools

font = {'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['ytick.major.width'] = 3


'''
read adsoprtion energy and barder charge from a csv file
'''
data = pd.read_csv('Ea_data.csv', header = 0)

metal = np.array(data['Metal'])
Ec = np.array(data['Ec_DFT'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])


'''
Prepare for the features based on the original data
'''
# Numerical orders
orders = [1, -1, 0.5, -0.5, 2, -2]

def transformers(xv, orders):

    '''
    Transform each column of primary featurs into secondary features
    '''
    x_features = np.reshape(xv, (len(xv),1))
    
    for oi in orders[1:]:
        x_features = np.concatenate((x_features, np.reshape(xv, (len(xv),1))**oi), axis = 1)
    
    '''
    Add additional features
    '''
    x_features = np.concatenate((x_features, np.log(np.reshape(xv, (len(xv),1)))), axis = 1)
    
    return x_features
'''
Get the names and orders
'''    
# primary and secondary feature names
x_primary_feature_names = ['Ec', 'Ebind']
x_secondary_feature_names_2d = []
orders_log = orders + ['ln']

# The number for all numerical opeators 
all_orders_log = []

for xi in x_primary_feature_names:  
    x_secondary_feature_names_2d.append([xi + '_' + str(oi) for oi in orders_log])
    all_orders_log += orders_log
    
x_secondary_feature_names = []
for xi in x_secondary_feature_names_2d:
    x_secondary_feature_names += xi


'''
Apply to the data
''' 
Ec_features = transformers(Ec, orders)    
Ebind_features = transformers(Ebind, orders)   


X_init = np.concatenate((Ec_features, Ebind_features),axis = 1) 


poly = PolynomialFeatures(2, interaction_only=True)
X_poly = poly.fit_transform(X_init)
orders_m = poly.powers_

'''
Select nonzero features
'''
x_features_poly = ['b'] # nonzero feature names in a 2d list
poly_indices_nonrepeat = [0] # indices in the polynominal order matrix 
x_plot_feature_names = ['b'] + x_secondary_feature_names
n_features = len(x_plot_feature_names)


'''
Get the indices and feature names for nonzero features, 
namely with nonzero order
'''
for pi, powers in enumerate(poly.powers_):
    
    powers_nonzero = (powers > 0).nonzero()[0]
    
    if not list(powers_nonzero) == []:
        
        features_nonzero = [x_secondary_feature_names[pi] for pi in powers_nonzero]
        orders_nonzero = np.array([all_orders_log[pi] for pi in powers_nonzero])
    
        try: 
            # making sure the zero sum is from one feature each
            ordersum = orders_nonzero.sum()
            f1_order = powers_nonzero[0] in range(0, len(x_secondary_feature_names_2d[0]))
            f2_order = powers_nonzero[1] in range(len(x_secondary_feature_names_2d[0]), len(x_secondary_feature_names_2d[0])+len(x_secondary_feature_names_2d[1]))
        
            if ordersum == 0:
                if (f1_order and f2_order):
                    pass
                else:
                    x_features_poly.append(features_nonzero)
                    poly_indices_nonrepeat.append(pi)
            else:
                    x_features_poly.append(features_nonzero)
                    poly_indices_nonrepeat.append(pi)
       
        except:
            
            x_features_poly.append(features_nonzero)
            poly_indices_nonrepeat.append(pi)

poly_indices_nonrepeat = np.array(poly_indices_nonrepeat)


x_features_poly_combined = []
for fi in x_features_poly:
    if len(fi) > 1:
        fi_combined = []
        for fj in fi:
            fi_combined += fj
            fi_combined = ''.join(fi_combined)

        x_features_poly_combined.append(fi_combined)
        
    else: x_features_poly_combined.append(fi[0])
        
'''
Process X and y, scale
'''
X = X_poly[:,poly_indices_nonrepeat]
y = Ea
Xscaler = StandardScaler().fit(X[:,1:])

sv = Xscaler.scale_
mv = Xscaler.mean_


X[:,1:] = Xscaler.transform(X[:,1:])
fit_int_flag = False # Not fitting for intercept, as the first coefficient is the intercept


#%% Preparation before regression
'''
Cross validation setting
'''
# Train test split, save 10% of data point to the test set
X_train, X_test, y_train, y_test, X_init_train, X_init_test = train_test_split(X, y, X_init, test_size=0.2, random_state=0)
                    
                    
# The alpha grid used for plotting path
alphas_grid = np.logspace(0, -3, 20)

# Cross-validation scheme                                  
rkf = RepeatedKFold(n_splits = 10, n_repeats = 10 , random_state = 0)


# Explicitly take out the train/test set
X_cv_train, y_cv_train, X_cv_test, y_cv_test = [],[],[],[]

for train_index, test_index in rkf.split(X_train):
    X_cv_train.append(X_train[train_index])
    y_cv_train.append(y_train[train_index])
    X_cv_test.append(X_train[test_index])
    y_cv_test.append(y_train[test_index])
    
    
#%% LASSO regression
'''   
# LassoCV to obtain the best alpha, the proper training of Lasso
'''
model_name = 'lasso_v2'
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
lasso_r2_train = r2_score(y_train, y_predict_train)


##Use alpha grid prepare for lassopath
lasso_RMSE_path, lasso_coef_path = rtools.cal_path(alphas_grid, Lasso, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
##lasso_path to get alphas and coef_path, somehow individual CV does not work
#lasso_alphas, lasso_coef_path, _ = lasso_path(X_train, y_train, alphas = alphas_grid, fit_intercept=fit_int_flag)
rtools.plot_path(X, y, lasso_alpha, alphas_grid, lasso_RMSE_path, lasso_coef_path, lasso_cv, model_name, output_dir)
lasso_RMSE, lasso_r2 = rtools.parity_plot(y, lasso_cv.predict(X), model_name, output_dir)


# The indices for non-zero coefficients/significant cluster interactions 
J_index = np.nonzero(lasso_coefs)[0]
# The number of non-zero coefficients/significant cluster interactions  
n_nonzero = len(J_index)
# The values of non-zero coefficients/significant cluster interactions  
J_nonzero = lasso_coefs[J_index] 

# collect nonzero freature names
x_feature_nonzero_combined = [x_features_poly_combined[pi] for pi in J_index]
x_feature_nonzero = [x_features_poly[pi] for pi in J_index]
rtools.plot_coef(J_nonzero, model_name, output_dir, x_feature_nonzero_combined)

'''
Convert the coefficient to unnormalized form
'''
lasso_coefs_unnormailized = np.zeros_like(lasso_coefs)
lasso_coefs_unnormailized[1:] = lasso_coefs[1:]/sv
lasso_coefs_unnormailized[0] = lasso_coefs[0] - np.sum(mv/sv*lasso_coefs[1:])


#%% Plot coefficients function
def make_coef_matrix(x_feature_nonzero, J_nonzero):
    
    '''
    Put the coefficient matrix back
    '''
    
    lasso_coef_matrix = np.zeros((n_features, n_features))
    
    for xi, feature_names in enumerate(x_feature_nonzero):
        Ji = J_nonzero[xi]
        
        if len(feature_names) == 1:
            
            if feature_names == 'b':
                lasso_coef_matrix[0,0] = Ji
                
            else:
                # row number
                ri = np.where(np.array(x_secondary_feature_names) == feature_names)[0][0] + 1
                # column number
                ci = 0
                lasso_coef_matrix[ri,ci] = Ji
                
        if len(feature_names) == 2:
            # row number
            ri = np.where(np.array(x_secondary_feature_names) == feature_names[1])[0][0] + 1
            ci = np.where(np.array(x_secondary_feature_names) == feature_names[0])[0][0] + 1
            
            lasso_coef_matrix[ri,ci]
            
    return lasso_coef_matrix


def plot_tri_correlation_matrix(coef_matrix, model_name, output_dir):
    
    '''
    Plot the correlation matrix in a lower trianglar fashion
    '''
    corr = coef_matrix.copy()
    
    # create mask, true for white, false to show the value
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    mask[0,0] = False
    mask[2,1] = True
    mask[4,3] = True
    mask[6,5] = True
    mask[9,8] = True
    mask[11,10] = True
    mask[13,12] = True
    
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 20))
    # Generate a custom diverging colormap
    cmap = sns.color_palette("RdBu_r", 7) 
    sns.set_style("white")
    sns.heatmap(corr, mask = mask, cmap=cmap, vmin = -0.5, vmax=0.5, center=0,
                square=True, linewidths=1.5, cbar_kws={"shrink": 0.7})
    ax.tick_params('both', length=0, width=0, which='major')
    ax.set_xticks(np.arange(0,len(x_plot_feature_names))+0.5)
    ax.set_xticklabels(x_plot_feature_names, rotation = 0)
    ax.set_yticks(np.arange(0,len(x_plot_feature_names))+0.5)
    ax.set_yticklabels(x_plot_feature_names, rotation = 360)
    ax.set_xlabel('Descriptor 1')
    ax.set_ylabel('Descriptor 2')
    fig.savefig(os.path.join(output_dir, model_name + '_coef_heatmap.png'))


#%% Ridge regression
'''
# Ridge regression
'''
'''
# RidgeCV to obtain the best alpha, the proper training of ridge
'''
model_name = 'ridge'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

alphas_grid_ridge = np.logspace(0, -3, 20)
ridgeCV = RidgeCV(alphas = alphas_grid_ridge,  cv = rkf, fit_intercept=fit_int_flag)
ridgeCV.fit(X_train, y_train)
ridge_alpha = ridgeCV.alpha_ 
ridge_intercept = ridgeCV.intercept_ 
ridge_coefs = ridgeCV.coef_

# Access the errors 
y_predict_test = ridgeCV.predict(X_test)
y_predict_train = ridgeCV.predict(X_train)

ridge_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
ridge_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
ridge_r2_train = r2_score(y_train, y_predict_train)   

# plot the rigde path
ridge_RMSE_path, ridge_coef_path = rtools.cal_path(alphas_grid_ridge, Ridge, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
rtools.plot_RMSE_path(ridge_alpha, alphas_grid_ridge, ridge_RMSE_path, model_name, output_dir)
rtools.plot_performance(X, y, ridgeCV,model_name, output_dir)

ridge_RMSE, ridge_r2 = rtools.parity_plot(y, ridgeCV.predict(X), model_name, output_dir)
rtools.plot_coef(ridgeCV.coef_, terms, model_name, output_dir)

ridge_coefs_unnormailized = np.zeros_like(ridge_coefs)
ridge_coefs_unnormailized[1:] = ridge_coefs[1:]/sv
ridge_coefs_unnormailized[0] = ridge_coefs[0] - np.sum(mv/sv*lasso_coefs[1:])






