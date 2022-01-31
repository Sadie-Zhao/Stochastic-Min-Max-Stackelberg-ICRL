from importlib.metadata import distribution
from stochasticAlg import create_states
import stochsticLibrary as lib
import numpy as np
from state import State

def project_to_bugdet_set(X, p, b):
    X_prec = X
    while (True): 
        X -= ((X @ p - b).clip(min= 0)/(np.linalg.norm(p)**2).clip(min= 0.01) * np.tile(p, reps = (b.shape[0], 1)).T).T
        X = X.clip(min = 0)
        if(np.linalg.norm(X - X_prec) <= 0.0002):
            break
        # print(f"Current iterate {X}\nPrevious Iterate {X_prec}")
        X_prec = X
    return X
    

demands = np.random.rand(3,2)+100
p = np.random.rand(2)+10
b = np.random.rand(3)

x_after = project_to_bugdet_set(demands, p.T,b.T)
print(b-p @ demands.T)
print(x_after)
print("after:", b- p @ x_after.T)

