# Training Scaling Relationships for Single-atom Catalysts
The notebooks contain the workflow of training scaling relationships based on physical descriptors (features obtained from density functional theory calculations) using various machine learning methods.

The scaling relationships are useful for fast prediction of desired properties and catalyst material screening, saving computing time by not doing quantumn calculations. 

## Developers
- Yifan Wang (wangyf@udel.edu)

## Scaling Relationships are developed for 
- Ebind, the binding energy of single-metal atom on a support
- Ea, the activation barrier for metal atom diffusion 

## Related Publication 
Su, Y.; Zhang, L.; __Wang, Y.__; Liu, J.; Muravev, V.; Alexopoulos, K.; Filot, A. W.; Vlachos, D. G.; Hensen, E. J. M. Stability of Heterogeneous Single-Atom Catalysts : A Scaling Law Mapping Thermodynamics to Kinetics (2019). (Submitted)

## Machine Learning Methods Used:
- LASSO regression
- Ridge regression
- Elastic net
- Ordinary Least Square (OLS) regression
- Genetic Programming (GP) based on sybomlic regression

## Getting Started 
- Train_Ea: the training for Ea
- Train_Ebind: the training for Ebind
- GP: files for training genetic programming models

## Dependencies 
- [Numpy](https://numpy.org/): Used for vector and matrix operations
- [Matplotlib](https://matplotlib.org/): Used for plotting
- [Scipy](https://www.scipy.org/): Used for linear algebra calculations
- [Pandas](https://pandas.pydata.org/): Used to import data from Excel files
- [Sklearn](https://scikit-learn.org/stable/): Used for training machine learning models
- [Seaborn](https://seaborn.pydata.org/): Used for plotting
- [Gplearn](https://gplearn.readthedocs.io/en/stable/): Used for training genetic programming models 


