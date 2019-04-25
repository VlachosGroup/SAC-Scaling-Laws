# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:09:01 2019

@author: yifan
"""

'''
regression plot tools
'''
import os
import numpy as np
import json
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
font = {'family' : 'normal', 'size'   : 12}
matplotlib.rc('font', **font)


#%% User define functions 



'''
Plot the regression results
'''
      
def predict_y(x, intercept, J_nonzero):
    
    # x is the column in pi matrix or the pi matrix 
    y = np.dot(x, J_nonzero) + intercept
    # the results should be the same as y = lasso_cv.predict(X)
    return y


def cal_path(alphas, model, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag):
    
    '''
    Calculate both RMSE and number of coefficients path for plotting purpose
    '''
    
    RMSE_path = []
    coef_path = []
    
    for j in range(len(X_cv_train)):
        
        test_scores = np.zeros(len(alphas))
        coefs_i = np.zeros(len(alphas))
        
        print('{} % done'.format(100*(j+1)/len(X_cv_train)))
        
        for i, ai in enumerate(alphas):
            
            estimator = model(alpha = ai,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state = 0)
            estimator.fit(X_cv_train[j], y_cv_train[j])
            # Access the errors, error per cluster
            test_scores[i] = np.sqrt(mean_squared_error(y_cv_test[j], estimator.predict(X_cv_test[j]))) #RMSE
            coefs_i[i] = len(np.nonzero(estimator.coef_)[0])
        
        RMSE_path.append(test_scores)
        coef_path.append(coefs_i)
    
    RMSE_path = np.transpose(np.array(RMSE_path))
    coef_path = np.transpose(np.array(coef_path))

    
    return RMSE_path, coef_path



def plot_coef_path(alpha, alphas, coef_path, model_name, output_dir = os.getcwd()):
    '''
    #plot alphas vs the number of nonzero coefficents along the path
    '''


    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), coef_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(coef_path, axis = 1), 
             label='Average across the folds', linewidth=2)     
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    plt.legend(frameon=False, loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("Number of Nonzero Coefficients ")    
    plt.tight_layout()
    

    fig.savefig(os.path.join(output_dir, model_name + '_a_vs_n.png'))
    #plt.show() 



def plot_RMSE_path(alpha, alphas, RMSE_path, model_name, output_dir = os.getcwd()):
        
    '''
    #plot alphas vs RMSE along the path
    '''

    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), RMSE_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(RMSE_path, axis = 1), 
             label='Average across the folds', linewidth=2)  
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    
    plt.legend(frameon=False,loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("RMSE/cluster(ev)")    
    plt.tight_layout()
   
    fig.savefig(os.path.join(output_dir, model_name  + '_a_vs_cv.png'))
    #plt.show()   
       
def plot_path(X, y, alpha, alphas, RMSE_path, coef_path, model, model_name, output_dir = os.getcwd()):
    
    '''
    Overall plot function for lasso/elastic net
    '''
    
    plot_coef_path(alpha, alphas, coef_path, model_name, output_dir)
    plot_RMSE_path(alpha, alphas, RMSE_path, model_name, output_dir)
    
    '''
    #make performance plot
    '''
    plot_performance(X, y, model, model_name, output_dir)
    


def plot_ridge_path(alpha, alphas, RMSE_path, model_name, output_dir = os.getcwd()):
    
    fig = plt.figure(figsize=(6, 4))
    
    #plt.plot(-np.log10(alphas), np.log10(RMSE_path), ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(RMSE_path, axis = 1), 
             label='Average across the folds', linewidth=2)  
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    
    plt.legend(frameon=False,loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("RMSE/cluster(ev)")    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name +'_a_vs_cv.png'))
    #plt.show()   

    
    
def plot_performance(X, y, model, model_name, output_dir = os.getcwd()): 
    
    '''
    #plot parity plot
    '''
    y_predict_all = model.predict(X)
    #y_predict_all = predict_y(pi_nonzero, intercept, J_nonzero)
    
    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, y_predict_all, s=60, facecolors='none', edgecolors='r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Predicted Cluster Energy (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name + '_parity.png'))
    #plt.show()
    
    '''
    #plot error plot
    '''

    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y,y_predict_all - y, s = 20, color ='r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Error Energy (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, np.zeros(len(lims)), 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    #ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name +'_error.png'))
    #plt.show()
    
    '''
    #plot error plot per atom
    '''

    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, (y_predict_all - y), s=20, color = 'r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Error Energy per atom (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, np.zeros(len(lims)), 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    #ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name + '_error_atom.png'))
    #plt.show()

def cal_performance(X, y, model, model_name): 
    
    y_predict_all = model.predict(X)
    RMSE = np.sqrt(mean_squared_error(y, y_predict_all))
    r2 = r2_score(y, y_predict_all)
    
    return RMSE, r2
    
    