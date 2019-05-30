from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (LeaveOneOut, RepeatedKFold,
                                     cross_val_score, train_test_split)
import pandas as pd
import graphviz

ri = 0
'''
read adsoprtion energy and barder charge from a csv file
'''
model_name = 'gp_Ea'
data = pd.read_csv('Ea_data.csv', header=0)

metal = np.array(data['Metal'])
Ec = np.array(data['Ec_DFT'])
Ebind = np.array(data['Ebind'])
Ea = np.array(data['Ea'])

X_init = np.stack((Ec, Ebind), 1)
X = X_init.copy()
y = Ea

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

est_gp = SymbolicRegressor(population_size=5000, metric = 'rmse', n_jobs = 5,
                           generations=20, stopping_criteria=0.1,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=ri)
est_gp.fit(X_train, y_train)
print(est_gp._program)

y_test_pred = est_gp.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print('Test {}: \n mae: {} \n rmse: {} \n'.format(model_name, test_mae, test_rmse))

dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ea_'+str(ri), format='png', cleanup=True)
