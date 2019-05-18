# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 12:40:43 2018

@author: yifan
"""


from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict

import pandas as pd
import numpy as np
import pickle 

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import matplotlib

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
data = pd.read_csv('Ea_data.csv', header = 0)

metal = np.array(data['Metal'])
Ebulk = np.array(data['Ebulk'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])


#%% PCA data
X = np.stack((Ebulk, Ebind, Ebind/Ebulk)    ,1) #, Ebind**2/Ebind
y = Ea

descriptors = ['Ebulk', 'Ebind', 'Ebind/Ebulk'] #, 'Ebulk^2/Ebind']
metal_types = np.unique(metal)
cm = ['r', 'b', 'purple', 'k', 'g', 'orange', 'pink', 'cyan', 'lightgreen']

#%% PCA parameters

nDescriptors = X.shape[1]
# select the number of PCs to plot in the bar graph
pc = 3
# select the number of PCs to plot in regression
pc_reg = 3


def plot_discriptors_st():    
    '''
    Plot the trend for each discriptor with site type color coded
    '''
    
    for cnt in range(nDescriptors):
        plt.figure(figsize=(6, 4))
        for metal_type, col in zip(metal_types, cm):
            indices = np.where(np.array(metal) == metal_type)[0]
            plt.scatter(X[:,cnt][indices],
                        y[indices],
                        label=metal_type,
                        facecolor = col, 
                        alpha = 1,
                        s  = 60)
            
            plt.xlabel(descriptors[cnt])
            plt.ylabel('Ea (eV)')
            
        plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
        plt.tight_layout()
        plt.show() 
            
plot_discriptors_st()            
#%% PCA 
# Normalize the data
X_std = StandardScaler().fit_transform(X)
# Covriance matrix of original data
cov_mat = np.cov(X_std.T) 

# PCA use sklearn
pca = PCA()    
Xpc = pca.fit_transform(X_std) 
# Covriance matrix from PCA
cov_pc = np.cov(Xpc.T) 


# Plot Covariance structure 
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,6))
ax1.set_title('X')
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(descriptors)
ax1.set_yticks([0,1,2])
ax1.set_yticklabels(descriptors)
im1 = ax1.imshow(cov_mat)
fig.colorbar(im1, ax = ax1, shrink = 0.5)

ax2.set_title('X_PCA')
im2 = ax2.imshow(cov_pc)
ax2.set_xticks([0,1,2])
ax2.set_xticklabels(descriptors)
ax2.set_yticks([0,1,2])
ax2.set_yticklabels(descriptors)
fig.colorbar(im2, ax = ax2, shrink = 0.5)

#eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_vals = pca.explained_variance_ #eigenvalues 
eig_vecs = pca.components_  # eigenvector
#tot = sum(eig_vals)
#var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
var_exp = pca.explained_variance_ratio_ #explained variance ratio
cum_var_exp = np.cumsum(var_exp) #cumulative variance ratio
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)

                      

plt.figure(figsize=(6, 4))

plt.bar(range(nDescriptors), var_exp, alpha=0.5, align='center',
        label='individual variance')
plt.step(range(nDescriptors), cum_var_exp, where='mid',
         label='cumulative variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xticks(np.arange(nDescriptors), 
            ['PC %i'%(w+1) for w in range(nDescriptors)])
plt.legend(loc='best', frameon=False)
plt.tight_layout()

    
#%% Plot the normalized desciptor loading

ind = 0
yvals = []
ylabels = []
bar_vals = []
space = 0.3

cm_bar = ['r', 'coral', 'pink',  'orange', 'y', 'gold', 'lightblue', 'lime', 'grey', 'green', 'brown'][:len(descriptors)]
fig = plt.figure(figsize=(10,6))


ax = fig.add_subplot(111)
n = len(descriptors)
width = (1 - space) / (len(descriptors))
indeces = np.arange(0, pc) + 0.5  

# Create a set of bars at each position
for i, pci in enumerate(eig_vecs[:pc]):
    
    vals = pci/np.sum(np.absolute(pci))
    
    pos = width*np.arange(n) + i 
    ax.bar(pos, vals, width=width, label=str(i+1), color = cm_bar) 
        
linex = np.arange(np.arange(0, pc).min() -0.5  , np.arange(0, pc).max()+ 2)

ax.set_xticks(indeces)
ax.set_xticklabels(list(np.arange(0,pc)+1))
ax.set_ylabel("Normalized Descriptoor Loading")
ax.set_xlabel("Principal Component #")    
  
# Add legend using color patches
patches = []
for c in range(n):
    patches.append(mpatches.Patch(color=cm_bar[c]))
plt.legend(patches, descriptors,
           bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)

plt.plot(linex, linex*0, c = 'k', lw = 0.8)
plt.show()

#%% Regression
def parity_plot_st(yobj,ypred, method = 'PCA'):
    '''
    Plot the parity plot of y vs ypred
    return R2 score and MSE for the model
    colorcode different site types
    '''
    MSE = mean_squared_error(yobj, ypred)
    score = r2_score(yobj, ypred)
    fig, ax = plt.subplots()
    for metal_type, col in zip(metal_types, cm):
        indices = np.where(np.array(metal) == metal_type)[0]
        ax.scatter(yobj[indices],
                    ypred[indices],
                    label=metal_type,
                    facecolor = col, 
                    alpha = 0.5,
                    s  = 60)
    ax.plot([yobj.min(), yobj.max()], [yobj.min(), yobj.max()], 'k--', lw=2)
    ax.set_xlabel('Objective')
    ax.set_ylabel('Predicted')
    plt.title(r'Method-{}, MSE-{:.2}, $r^2$ -{:.2}'.format(method, MSE, score))
    plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
    plt.show()
    
    return MSE, score

def error_distribution(yobj, ypred, method  = 'PCA'):
    
    '''
    Plot the error distribution
    return the standard deviation of the error distribution
    '''
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(yobj - ypred,density=1, alpha=0.5, color='steelblue')
    mu = 0
    sigma = np.std(yobj - ypred)
    x_resid = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x_resid,norm.pdf(x_resid, mu, sigma), color='r')
    plt.title(r'Method-{}, $\sigma$-{:.2}'.format(method, sigma))
    
    return sigma

def linear_regression(degree):
    '''
    # Create linear regression object
    '''
    return Pipeline([("polynomial_features", PolynomialFeatures(degree=degree,
                                                                include_bias=False)),
                     ("linear_regression", linear_model.LinearRegression())]
                    )

Xreg = Xpc[:,:pc_reg]
degree = 2


estimator  = linear_regression(degree)
estimator.fit(Xreg, y)

regr_pca = estimator.named_steps['linear_regression']
coefs = regr_pca.coef_
intercept = regr_pca.intercept_
poly = estimator.named_steps['polynomial_features']
feature_names = []
for i in range(pc_reg): feature_names.append('x'+ str(i+1))
terms = poly.get_feature_names(feature_names)

y_pca = estimator.predict(Xreg)
mse_pca, score_pca = parity_plot_st(y, y_pca)
sigma_pca = error_distribution(y, y_pca)

'''
Plot the magnitude of each parameters
'''
xi = np.arange(len(coefs))
fig, ax = plt.subplots()
plt.bar(xi, coefs)
linex = np.arange(xi.min()-1, xi.max()+2)
plt.plot(linex, linex*0, c = 'k')
plt.xticks(xi, terms, rotation=45, fontsize = 8 )
plt.ylabel("Regression Coefficient Value (eV)")
plt.xlabel("Regression Coefficient")  
plt.show()

#%% Cross-validation
y_cv = cross_val_predict(estimator, Xreg, y, cv=10)
mse_cv, score_cv = parity_plot_st(y, y_cv)

#%% Compare two models
y_model2 = 0.6185*Ebind**2/Ebulk - 0.1477
mse_model2 = mean_squared_error(y, y_model2)
score_model2 = r2_score(y, y_model2)
fig, ax = plt.subplots()

ax.scatter(y, y_pca, label='PC Regression', facecolors='r', alpha = 0.5, s  = 60)
ax.scatter(y, y_model2, label='Universal Scaling', facecolors='b', alpha = 0.5, s  = 60)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('DFT-Calculated ')
ax.set_ylabel('Model Prediction')
#plt.title(r'Method-{}, MSE-{:.2}, $r^2$ -{:.2}'.format(method, MSE, score))
plt.legend(loc= 'lower right', frameon=False)
plt.show()
