# Training Scaling Relationships for Single-atom Catalysts
The notebooks contain the workflow of training scaling relationships based on physical descriptors (features obtained from density functional theory calculations) using various machine learning methods.

## Developers
- Yifan Wang (wangyf@udel.edu)

## Scaling Relationships are developed for 
- $E_{bind}$, the binding energy of single-metal atom on a support
- $E_a$, the activation barrier for metal atom diffusion 

## Related Publication 
Su, Y.; Zhang, L.; __Wang, Y.__; Liu, J.; Muravev, V.; Alexopoulos, K.; Filot, A. W.; Vlachos, D. G.; Hensen, E. J. M. Stability of Heterogeneous Single-Atom Catalysts : A Scaling Law Mapping Thermodynamics to Kinetics. (Submitted)

## Machine Learning Methods Used:
- LASSO regression
- Ridge regression
- Elastic net
- Ordinary Least Square (OLS) regression
- Genetic Programming based on sybomlic regression

## Dependencies 
- Numpy
- Matplotlib
- Scipy
- Pandas
- Sklearn
- Seaborn


