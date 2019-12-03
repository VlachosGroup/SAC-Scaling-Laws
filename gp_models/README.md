# Training a genetic programming model based on symbolic regression

## Definition 
Genetic programming (GP) with symbolic regression is a supervised learning technique to identify an underlying mathematic expression that best describes the relationship between input and output data. The search space consists of combinations of simple mathematical operators on the input descriptors. An evolutionary algorithm is used to evolve a population of randomly generated candidate models according to natural-selection rules (selection, crossover and mutation). Each model is associated with a fitness value, which in our case is the RMSE value of diffusion barrier Ea. The advantage of GP is that no manual combination of descriptors is needed. 

## Useful Files
train_GP_Ea.py trains the model to predict Ea based on two physical descriptors (Ebind and Ec).

The training is repeated 5 times with different random seeds. The model forms are saved in Ea_GP_models.xlsx and the graphic represenation of the syntax trees can be found in the folder trees.
