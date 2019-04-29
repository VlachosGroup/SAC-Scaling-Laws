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





font = {'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 16
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 16
matplotlib.rcParams['ytick.major.width'] = 3


'''
read adsoprtion energy and barder charge from a csv file
'''
data = pd.read_csv('Ea_data.csv', header = 0)

metal = np.array(data['Metal'])
Ec = np.array(data['Ebulk'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])


X_init = np.stack((Ec, Ebind, Ebind/Ec, Ec/Ebind)  ,1) 
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_init)
#%%

pi = poly.powers_
pi_unrepeated =  [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14]
X_poly_unrepeated = X_poly[:, np.array(pi_unrepeated)]
X = X_poly_unrepeated
y = Ea
fit_int_flag = False # Not fitting for intercept, as the first coefficient is the intercept
terms = ['$b_0$', '$E_c$', '$E_{bind}$', r'$\frac{E_{bind}}{E_c}$', r'$\frac{E_c}{E_{bind}}$', r'$E_c^2$', '$E_cE_{bind}$', r'$\frac{E_c^2}{E_{bind}}$', '$E_{bind}^2$', r'$\frac{E_{bind}^2}{E_c}$', r'$\frac{E_{bind}^2}{E_c^2}$', r'$\frac{E_c^2}{E_{bind}^2}$']

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
from sklearn.linear_model import LassoCV, lasso_path, Lasso

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
lasso_RMSE, lasso_r2 = rtools.parity_plot(y, lasso_cv.predict(X), model_name, output_dir)
rtools.plot_coef(lasso_cv.coef_, terms, model_name, output_dir)

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
model_name = 'ridge'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

alphas_grid_ridge = np.logspace(0, -3, 20)
ridgeCV = RidgeCV(alphas = alphas_grid_ridge,  cv = rkf, fit_intercept=fit_int_flag)
ridgeCV.fit(X_train, y_train)
ridge_alpha = ridgeCV.alpha_ 
ridge_intercept = ridgeCV.intercept_ 

# Access the errors 
y_predict_test = ridgeCV.predict(X_test)
y_predict_train = ridgeCV.predict(X_train)

ridge_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
ridge_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))


# plot the rigde path
ridge_RMSE_path, ridge_coef_path = rtools.cal_path(alphas_grid_ridge, Ridge, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
rtools.plot_RMSE_path(ridge_alpha, alphas_grid_ridge, ridge_RMSE_path, model_name, output_dir)
rtools.plot_performance(X, y, ridgeCV,model_name, output_dir)

ridge_RMSE, ridge_r2 = rtools.parity_plot(y, ridgeCV.predict(X), model_name, output_dir)
rtools.plot_coef(ridgeCV.coef_, terms, model_name, output_dir)


#%%

from sklearn.linear_model import ElasticNetCV, enet_path, ElasticNet

model_name = 'enet'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

  
def l1_enet(ratio):
    
    '''
    input l1 ratio and return the model, non zero coefficients and cv scores
    training elastic net properly
    '''
    enet_cv  = ElasticNetCV(cv = rkf, l1_ratio=ratio,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state=5)
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
enet05_RMSE, enet05_r2 = rtools.parity_plot(y, enet05.predict(X), model_name, output_dir)
rtools.plot_coef(enet05.coef_, terms, model_name, output_dir)

#%%
'''
PLS regression 
'''
from sklearn.cross_decomposition import PLSRegression

PLS = PLSRegression(n_components = 3, tol=1e-8) #<- N_components tells the model how many sub-components to select
PLS.fit(X_train,y_train) 

# Access the errors 
y_predict_test = PLS.predict(X_test)
y_predict_train = PLS.predict(X_train)

PLS_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
PLS_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))

PLS_RMSE, PLS_r2 = rtools.cal_performance(X, y, PLS)

#%%
'''
Second order regression
'''
model_name = 'P2'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

P2 = linear_model.LinearRegression(fit_intercept=fit_int_flag)
P2.fit(X_train,y_train) 

# Access the errors 
y_predict_test = P2.predict(X_test)
y_predict_train = P2.predict(X_train)

P2_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
P2_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))

P2_RMSE, P2_r2 = rtools.parity_plot(y, P2.predict(X), model_name, output_dir)
rtools.plot_coef(P2.coef_, terms, model_name, output_dir)



#%%
'''
Univerisal scaling relation
'''
model_name = 'USR'
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

def USR(X_init):
    Ebind = X_init[:,1]
    Ec = X_init[:,0]
    Ea = Ebind**2/Ec *0.6185 -0.1477
    
    return Ea

X_USR_test = X_test[:,1:3]
X_USR_train = X_train[:,1:3]
X_USR = X[:,1:3]

# Access the errors 
y_predict_test = USR(X_USR_test)
y_predict_train = USR(X_USR_train)

USR_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
USR_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
USR_RMSE, USR_r2 =rtools.parity_plot(y, USR(X_USR), model_name, output_dir)
#USR_sigma = rtools.error_distribution(y, USR(X_USR), model_name)
#USR_RMSE, USR_r2 = rtools.cal_performance(X, y, LM1)


#%%
'''
Compare enet 05 and USR
'''



fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y, enet05.predict(X), label='Elastic Net ($R^2$ = 0.965)', facecolors='r', alpha = 0.7, s  = 60)
ax.scatter(y, USR(X_USR), label='Universal Scaling ($R^2$ = 0.966)', facecolors='b', marker="o", alpha = 0.7, s  = 60)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('DFT-Calculated (eV)')
ax.set_ylabel('Model Prediction (eV)')
#plt.title(r'Method-{}, MSE-{:.2}, $r^2$ -{:.2}'.format(method, MSE, score))
plt.legend(loc= 'upper left', frameon=False)
plt.show()

'''
Compare different regression method
'''

regression_method = [ 'USM', 'Elastic Net', 'LASSO', 'Ridge', 'OLS']

means_test = np.array([ USR_RMSE_test, enet05_RMSE_test, lasso_RMSE_test, ridge_RMSE_test, P2_RMSE_test])

r2s = np.array([ USR_r2, enet05_r2, lasso_r2, ridge_r2, P2_r2])

base_line = 0
x_pos = np.arange(len(regression_method))
opacity = 0.8
bar_width = 0.25
fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()
rects2 = ax1.bar(x_pos, means_test - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='r',
                label='Test')
rects3 = ax2.bar(x_pos+bar_width, r2s - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='b',
                label='r2')
#plt.ylim([-1,18])
ax1.set_xticks(x_pos+bar_width/2)
ax1.set_xticklabels(regression_method, rotation=20)
ax1.set_xlabel('Model Name')
#plt.legend(loc= 'best', frameon=False)

ax1.set_ylabel('Testing RMSE (eV)', color = 'r')
ax1.set_ylim([0, 0.3])
ax1.tick_params('y', colors='r')

ax2.set_ylabel('$R^2$',color = 'b')
ax2.set_ylim([0.90, 1])
ax2.tick_params('y', colors='b')

