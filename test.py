from importlib.metadata import distribution
from stochasticAlg import create_states
import stochsticLibrary as lib
import numpy as np
from state import State

def gradient_of_log(states, current_state, s_i, prices, demands):
    saved_budgets = current_state.get_saved(prices, demands)
    diff_sum = np.sum( [np.linalg.norm(next_state.budgets - saved_budgets)**2 for next_state in states] )
    diff_si = np.linalg.norm(s_i.budgets - saved_budgets)**2
    vector_part_sum = 2 * np.sum( [next_state.budgets - saved_budgets for next_state in states], axis = 0 )
    vector_part_si = 2 * (s_i.budgets - saved_budgets)
    return (diff_sum - diff_si) * ((vector_part_sum - vector_part_si) @ demands) - diff_sum * (vector_part_sum @ demands)



num_goods = 3
num_buyers = 2
discount = 0.01
states = create_states(num_buyers, num_goods, 3, "linear", 10, 10, 100, 0, 10, 10)
current_state = states[0]
s_i = states[1]
prices = np.random.rand(num_goods)
demands = np.random.rand(num_buyers, num_goods)

num_states = len(states)
v= np.random.rand(num_states)
sum = np.zeros(num_goods)
for s_i in states:
    sum += v[s_i.index] * gradient_of_log(states, current_state, s_i, prices, demands)

discounted_exp_of_v = discount * 1/num_states * sum
print(discounted_exp_of_v)
