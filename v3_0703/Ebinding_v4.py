
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:20:23 2018

@author: wangyf
"""

'''
Predict Ebinding based on 'Ebulk', 'Evac', 'delta X' , 'CN'
Standize data
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
from matplotlib.pyplot import cm


font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 12
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 12
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['legend.fontsize'] = 16



'''
read adsoprtion energy and barder charge from a csv file
'''
data = pd.read_csv('Ea_data.csv', header = 0)

descriptors  = ['Ec', 'Evac', 'delta X' , 'CN', 'angle']
Ea = np.array(data['Ea'])
metal = np.array(data['metal'])
support = np.array(data['support'])


#%% PCA data
X_init = np.array(data[descriptors]) #, Ebind**2/Ebind
y = np.array(data['Ebind'])
poly = PolynomialFeatures(2)

X_poly = poly.fit_transform(X_init)
X = X_poly
'''
Take out the ones
'''



fit_int_flag = False # Not fitting for intercept, as the first coefficient is the intercept

X[:,1:] = StandardScaler().fit_transform(X[:,1:])

metal_types = np.unique(metal)
#cm = ['r', 'b', 'purple', 'k', 'g', 'orange', 'pink', 'cyan', 'lightgreen']

#%% PCA parameters

nDescriptors = X.shape[1]
# select the number of PCs to plot in the bar graph
#pc = len(descriptors)
## select the number of PCs to plot in regression
#pc_reg = 4

#
#def plot_discriptors_st():    
#    '''
#    Plot the trend for each discriptor with site type color coded
#    '''
#    
#    for cnt in range(nDescriptors):
#        plt.figure(figsize=(6, 4))
#        for metal_type, col in zip(metal_types, cm):
#            indices = np.where(np.array(metal) == metal_type)[0]
#            plt.scatter(X[:,cnt][indices],
#                        y[indices],
#                        label=metal_type,
#                        facecolor = col, 
#                        alpha = 1,
#                        s  = 60)
#            
#            plt.xlabel(descriptors[cnt])
#            plt.ylabel('Ea (eV)')
#            
#        plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
#        plt.tight_layout()
#        plt.show() 
#            
#plot_discriptors_st()            


#%% Preparation before regression
# random state
rs = 0
# Train test split, save 10% of data point to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = rs)
                       
# The alpha grid used for plotting path
alphas_grid = np.logspace(0, -3, 100)

# Cross-validation scheme                                  
rkf = RepeatedKFold(n_splits = 10, n_repeats = 10 , random_state = rs)


# Explicitly take out the train/test set
X_cv_train, y_cv_train, X_cv_test, y_cv_test = [],[],[],[]
for train_index, test_index in rkf.split(X_train):
    X_cv_train.append(X_train[train_index])
    y_cv_train.append(y_train[train_index])
    X_cv_test.append(X_train[test_index])
    y_cv_test.append(y_train[test_index])
    
    
#%% LASSO regression
from sklearn.linear_model import LassoCV, lasso_path, Lasso

'''   
# LassoCV to obtain the best alpha, the proper training of Lasso
'''
model_name = 'lasso_Ebind'
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

lasso_cv  = LassoCV(cv = rkf,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state= rs)
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
lasso_RMSE, lasso_r2 = rtools.parity_plot(y, lasso_cv.predict(X), model_name, output_dir, lasso_RMSE_test)
#rtools.plot_coef(lasso_cv.coef_, terms, model_name, output_dir)

# The indices for non-zero coefficients/significant cluster interactions 
J_index = np.nonzero(lasso_coefs)[0]
# The number of non-zero coefficients/significant cluster interactions  
n_nonzero = len(J_index)
# The values of non-zero coefficients/significant cluster interactions  
J_nonzero = lasso_coefs[J_index] 


#%%
'''
# Ridge regression
'''
from sklearn.linear_model import RidgeCV, Ridge
'''
# RidgeCV to obtain the best alpha, the proper training of ridge
'''
model_name = 'ridge_Ebind'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

alphas_grid_ridge = np.logspace(3, -3, 20)
ridgeCV = RidgeCV(alphas = alphas_grid_ridge,  cv = rkf, fit_intercept=fit_int_flag)
ridgeCV.fit(X_train, y_train)
ridge_alpha = ridgeCV.alpha_ 
ridge_intercept = ridgeCV.intercept_ 

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

ridge_RMSE, ridge_r2 = rtools.parity_plot(y, ridgeCV.predict(X), model_name, output_dir, ridge_RMSE_test)
#rtools.plot_coef(ridgeCV.coef_, terms, model_name, output_dir)


#%%

from sklearn.linear_model import ElasticNetCV, enet_path, ElasticNet

model_name = 'enet_Ebind'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

  
def l1_enet(ratio):
    
    '''
    input l1 ratio and return the model, non zero coefficients and cv scores
    training elastic net properly
    '''
    enet_cv  = ElasticNetCV(cv = rkf, l1_ratio=ratio,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state= rs)
    enet_cv.fit(X_train, y_train)
    
    # the optimal alpha
    enet_alpha = enet_cv.alpha_
    enet_coefs = enet_cv.coef_
    n_nonzero = len(np.where(abs(enet_coefs)>=1e-7)[0])
    # Access the errors 
    y_predict_test = enet_cv.predict(X_test)
    y_predict_train = enet_cv.predict(X_train)
    
    # error per cluster
    enet_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
    enet_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))


    return enet_cv, enet_alpha, n_nonzero, enet_RMSE_test, enet_RMSE_train

# The vector of l1 ratio
l1s = list(np.around(np.arange(0.1,1,0.05), decimals= 2))
#l1s = [.1, .5, .7, .9,  .95,  .99, 1]
#l1s = [0.95]

enet = []
enet_alphas = []
enet_n  = []
enet_RMSE_test = []
enet_RMSE_train = []
enet_RMSE_test_atom = []
enet_RMSE_train_atom = []


for i, l1i in enumerate(l1s):
    print('{} % done'.format(100*(i+1)/len(l1s)))
    enet_cv, ai, n, RMSE_test, RMSE_train = l1_enet(l1i)
    
    enet.append(enet_cv)
    enet_alphas.append(ai)
    enet_n.append(n)
    
    enet_RMSE_test.append(RMSE_test)
    enet_RMSE_train.append(RMSE_train)


#%% Save elastic net results to csv
import pandas as pd

# expand the vector, put the result of ridge to the first
l1_ratio_v = np.array([0] + l1s)
enet_n_v  = np.array([X.shape[1]] + enet_n)
enet_RMSE_test_v = np.array([ridge_RMSE_test] + enet_RMSE_test)

enet_RMSE_test_v = [ridge_RMSE_test] + enet_RMSE_test
fdata = pd.DataFrame(np.transpose([l1_ratio_v, enet_n_v, enet_RMSE_test_v]), columns = ['l1 ratio', 'number of cluster', 'error per cluster (eV)'])
fdata.to_csv(os.path.join(output_dir, 'enet_data.csv'), index=False, index_label=False)


#%% Plot elastic net results

plt.figure(figsize=(8,6))
fig, ax1 = plt.subplots()
ax1.plot(l1_ratio_v, enet_RMSE_test_v, 'bo-')
ax1.set_xlabel('L1 Ratio')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('RMSE/cluster(ev)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(l1_ratio_v, enet_n_v, 'r--')
ax2.set_ylabel('# Nonzero Coefficients', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'elastic_net.png'))



#%% Plot elastic net results
plt.figure(figsize=(8,6))
fig, ax1 = plt.subplots()
ax1.plot(l1_ratio_v, enet_RMSE_test_v, 'bo-')
ax1.set_xlabel('L1 Ratio')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('RMSE/cluster(ev)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(l1_ratio_v, enet_n_v, 'r--')
ax2.set_ylabel('Number of Nonzero Coefficients', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'elastic_net.png'))


'''
#Use alpha grid prepare for enet_path when l1_ratio  = 0.95
'''
enet05 = enet[l1s.index(0.5)]
enet05_RMSE_test = np.sqrt(mean_squared_error(y_test, enet05.predict(X_test)))
enet_RMSE_path_05, enet_coef_path_05 = rtools.cal_path(alphas_grid, ElasticNet, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)    
rtools.plot_path(X, y, enet_alphas[l1s.index(0.5)], alphas_grid, enet_RMSE_path_05, enet_coef_path_05, enet05, model_name, output_dir)

enet05_RMSE, enet05_r2 = rtools.cal_performance(X, y,enet05 )
enet05_r2_train = r2_score(y_train, y_predict_train)

# the optimal alpha from lassocv
enet05_alpha = enet05.alpha_
# Coefficients for each term
enet05_coefs = enet05.coef_
# The original intercepts 
enet05_intercept = enet05.intercept_


# The indices for non-zero coefficients/significant cluster interactions 
J_index = np.nonzero(enet05_coefs)[0]
# The number of non-zero coefficients/significant cluster interactions  
n_nonzero = len(J_index)
# The values of non-zero coefficients/significant cluster interactions  
J_nonzero = enet05_coefs[J_index] 
enet05_RMSE, enet05_r2 = rtools.parity_plot(y, enet05.predict(X), model_name, output_dir, enet05_RMSE_test)
#rtools.plot_coef(enet05.coef_, terms, model_name, output_dir)


#%%
'''
First order regression
'''
model_name = 'OLS_Ebind'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

OLS = linear_model.LinearRegression(fit_intercept=fit_int_flag)
OLS.fit(X_train,y_train) 

# Access the errors 
y_predict_test = OLS.predict(X_test)
y_predict_train = OLS.predict(X_train)

OLS_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
OLS_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
OLS_r2_train = r2_score(y_train, y_predict_train)

OLS_RMSE, OLS_r2 = rtools.parity_plot(y, OLS.predict(X), model_name, output_dir, OLS_RMSE_test)
#rtools.plot_coef(LM1.coef_, terms, model_name, output_dir)


#%% Parity plot
'''
Based on metal
'''

metal_types = np.unique(metal)
types = metal_types.copy()
category = metal.copy()

fig, ax = plt.subplots(figsize=(6, 6))
color_set = cm.jet(np.linspace(0,1,len(types)))
for type_i, ci in zip(types, color_set):
    indices = np.where(np.array(category) == type_i)[0]
    ax.scatter(y[indices],
                    OLS.predict(X)[indices],
                    label=type_i,
                    facecolor = ci, 
                    alpha = 0.8,
                    s  = 60)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--',  lw=2)
ax.set_xlabel('DFT-Calculated (eV) ')
ax.set_ylabel('Model Prediction (eV)')
plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)




'''
Based on support
'''
support_types = np.unique(support)
types = support_types.copy()
category = support.copy()

fig, ax = plt.subplots(figsize=(6, 6))
color_set = cm.jet(np.linspace(0,1,len(types)))
for type_i, ci in zip(types, color_set):
    indices = np.where(np.array(category) == type_i)[0]
    ax.scatter(y[indices],
                    OLS.predict(X)[indices],
                    label=type_i,
                    facecolor = ci, 
                    alpha = 0.8,
                    s  = 60)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--',  lw=2)
ax.set_xlabel('DFT-Calculated (eV) ')
ax.set_ylabel('Model Prediction (eV)')
plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)



#%%

'''
Compare different regression method
'''

regression_method = [ 'Elastic Net', 'LASSO', 'Ridge', 'OLS']
n_method = len(regression_method)

means_test = np.array([ enet05_RMSE_test, lasso_RMSE_test, ridge_RMSE_test, OLS_RMSE_test])
r2s = np.array([enet05_r2, lasso_r2, ridge_r2, OLS_r2])

#r2s = np.array([ enet05_r2, lasso_r2, ridge_r2, OLS_r2])

base_line = 0
x_pos = np.arange(len(regression_method))
opacity = 0.8
bar_width = 0.25
fig, ax1 = plt.subplots(figsize=(6,6))
ax2 = ax1.twinx()
rects2 = ax1.bar(x_pos, means_test - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='r',
                label='Test')
rects3 = ax2.bar(x_pos+bar_width, r2s - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='b',
                label='r2')
#plt.ylim([-1,18])
ax1.set_xticks(x_pos+bar_width/2)
ax1.set_xticklabels(regression_method, rotation=0)
ax1.set_xlabel('Model Name')
#plt.legend(loc= 'best', frameon=False)

ax1.set_ylabel('Testing RMSE (eV)', color = 'r')
ax1.set_ylim([0, 1])
ax1.tick_params('y', colors='r')

ax2.set_ylabel('$R^2$',color = 'b')
ax2.set_ylim([0.8, 1])
ax2.tick_params('y', colors='b')