# Training Scaling Relationships for Single-atom Catalysts
The notebooks contain the workflow of training scaling relationships based on physical descriptors (features obtained from density functional theory calculations) using various machine learning methods.

## Developers
- Yifan Wang (wangyf@udel.edu)

## Scaling Relationships are developed for 
- Ebind, the binding energy of single-metal atom on a support
- Ea, the activation barrier for metal atom diffusion 

## Related Publication 
Su, Y.; Zhang, L.; __Wang, Y.__; Liu, J.; Muravev, V.; Alexopoulos, K.; Filot, A. W.; Vlachos, D. G.; Hensen, E. J. M. Stability of Heterogeneous Single-Atom Catalysts : A Scaling Law Mapping Thermodynamics to Kinetics. (Submitted)

## Machine Learning Methods Used:
- LASSO regression
- Ridge regression
- Elastic net
- Ordinary Least Square (OLS) regression
- Genetic Programming (GP) based on sybomlic regression

# Getting Started 
- Train_Ea: the training for Ea
- Train_Ebind: the training for Ebind
- GP: files for training genetic programming models

## Dependencies 
- Numpy: Used for vector and matrix operations
- Matplotlib: Used for plotting
- Scipy: Used for linear algebra calculations
- Pandas: Used to import data from Excel files
- Sklearn: Used for training machine learning models
- Seaborn: Used for plotting
- Gplearn: Used for training genetic programming models 


