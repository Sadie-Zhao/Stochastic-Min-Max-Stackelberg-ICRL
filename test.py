from importlib.metadata import distribution
from stochasticAlg import create_states
import stochsticLibrary as lib
import numpy as np
from state import State

num_buyers = 3
num_goods = 4
A = [np.random.rand(num_goods) * 10 for i in range(10)]
demands = np.random.rand(num_buyers, num_goods)
demands_policy = [np.random.rand(num_buyers, num_goods) for index in range(3)]


file_name = 'resluts.npy'
with open(file_name, 'wb') as f:
    np.save(f, A)
    np.save(f, demands_policy)

with open(file_name, 'rb') as f:
    A = np.load(f)
    demands_policy = np.load(f)
    
    print(A)
    print(demands_policy)

