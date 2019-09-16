# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:31:43 2019

@author: yifan
"""

'''
Plot Ea vs Ebind
Ea vs Ec
Ea vs Ebind/Ec

'''


import os
import pickle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
 
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


from scipy.stats.stats import pearsonr


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

metal = np.array(data['metal'])
support = np.array(data['support'])
Ec = np.array(data['Ec'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])



#%% Parity plot

model_name = 'Ea_plots'
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    


def parity_type_plot(y1, y2, y2_label, variable_name,  types, category, legend_labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    color_set = cm.jet(np.linspace(0,1,len(types)))
    for type_i, ci, label_i in zip(types, color_set, legend_labels):
        indices = np.where(np.array(category) == type_i)[0]
        ax.scatter(y1[indices],
                        y2[indices],
                        label=label_i,
                        facecolor = ci, 
                        alpha = 0.8,
                        s  = 100)
    #ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--',  lw=2)
    ax.set_ylabel(r'$\rm E_{a}$ (eV) ')
    ax.set_xlabel(y2_label + ' (eV)')
    pr = pearsonr(y1, y2)[0]
    plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
    plt.text(min(y1), max(y2) - 0.1, r'$\rm r_{pearson}$ = ' + str(np.around(pr, decimals = 3))) 
    fig.savefig(os.path.join(output_dir, model_name + '_parity_support_'+ variable_name + '.png'))

'''
Based on support
'''
support_labels = [r'$\rm CeO_{2}(100)$', r'$\rm CeO_{2}(111)$', 'Graphene', 'MgO(100)', '2H-'+r'$\rm MoS_{2}(0001)$', r'$\rm SrTiO_{3}(100)$',
         'Steps of ' + r'$\rm CeO_{2}$', r'$\rm TiO_{2}(110)$', 'ZnO(100)']
support_types = np.unique(support)
parity_type_plot(Ebind, Ea, r'$\rm E_{bind}$', 'Ebind',support_types, support, support_labels)
parity_type_plot(Ec, Ea, r'$\rm E_{c}$', 'Ec', support_types, support, support_labels)
parity_type_plot(Ebind/Ec, Ea,  r'$\rm E_{bind}/E_c$', 'Ebind_Ec', support_types, support, support_labels)
#fig, ax = plt.subplots(figsize=(6, 6))
#color_set = cm.jet(np.linspace(0,1,len(types)))
#for type_i, ci in zip(types, color_set):
#    indices = np.where(np.array(category) == type_i)[0]
#    ax.scatter(y[indices],
#                    lasso_cv.predict(X)[indices],
#                    label=type_i,
#                    facecolor = ci, 
#                    alpha = 0.8,
#                    s  = 60)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--',  lw=2)
#ax.set_xlabel('DFT-Calculated (eV) ')
#ax.set_ylabel('Model Prediction (eV)')
#plt.legend(bbox_to_anchor = (1.02, 1),loc= 'upper left', frameon=False)
#plt.text(3, 1, '$RMSE_{test}$ = ' + str(np.around(lasso_RMSE_test, decimals = 3)))
#plt.text(4,0.4, '$R^2$ = ' + str(np.around(lasso_r2, decimals = 3)) )