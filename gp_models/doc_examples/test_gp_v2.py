import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import graphviz
from collections import OrderedDict

df = pd.read_csv('dataset_Facebook.csv', sep=';')

df.head()


feature_names = ['Category', 'Type', 'Post Month',
                 'Post Weekday', 'Post Hour', 'Paid']
target_name = 'Total Interactions'

X = df[feature_names]
y = df[target_name]


# manually pick categorical variables
categorical_names = ['Category', 'Type', 'Paid']
X.loc[:, categorical_names] = X[categorical_names].astype('object')
X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=42)


# random_state=42 (train_test_split and SymbolicRegressor) and generations=15 gives the same results as in blog
# Compare symbolic regression
#   ridge regression
#   random forest
models = {'sr': SymbolicRegressor(generations=15, verbose=4, max_samples=0.8, random_state=42),
          'lm': make_pipeline(StandardScaler(), RidgeCV()),
          'rf': RandomForestRegressor()}

for model_name, model_instance in models.items():
    print('Training model {}'.format(model_name))
    model_instance.fit(X_train, y_train)


# Evaluation
for model_name, model_instance in models.items():
    y_test_pred = model_instance.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    print('Model {}: \n mae: {} \n mse: {} \n'.format(model_name, mae, mse))

# Print fittest solution
print(models['sr']._program)

# Export to a graph instance
graph = models['sr']._program.export_graphviz()
graph_str = str(graph)
program_str = str(models['sr']._program)

# Replace X{} with actual features names
mapping_dict = {'X{}'.format(i): X.columns[i]
                for i in reversed(range(X.shape[1]))}

for old_value, new_value in mapping_dict.items():
    graph_str = graph_str.replace(old_value, new_value)
    program_str = program_str.replace(old_value, new_value)


# Save localy
src = graphviz.Source(graph_str)
src.render('result.gv', view=True)
