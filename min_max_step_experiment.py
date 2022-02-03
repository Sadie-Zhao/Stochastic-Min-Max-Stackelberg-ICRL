
from concurrent.futures import process
import stochasticAlg as alg
import stochsticLibrary as lib
import numpy as np
from state import State
import fisherSolver as solver
import matplotlib.pyplot as plt
import math






num_goods = 3
num_buyers = 2
num_states = 5
discount = 0.01
states = alg.create_states(num_buyers, num_goods, num_states, "linear", 10, 10, 100, 0, 10, 10)
current_state = states[0]
s_i = states[1]
# prices = np.random.rand(num_goods) + 10000
# demands = np.random.rand(num_buyers, num_goods)
# lambdas = np.random.rand(num_buyers) + 10000

# learning_rates = [0.07,0.01,0.001]   ##### Fix 0.01 and 0.01 for demands and lambdas!!
learning_rates = [0.07,0.01,0.001]   ##### Fix 0.01 and 0.01 for demands and lambdas!!

v = [1,3,2,41,2,5,6,32,10,2]

prices, demands, final_q = alg.min_max_Q(num_buyers, num_goods, current_state, states, v, discount, 250, 100, learning_rates)

print(final_q)
