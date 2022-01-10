import stochsticLibrary as lib
import numpy as np
from state import State

num_states = 10
# Assigned probbaility of transition between states
assigned_prob = np.random.rand(num_states, num_states)
assigned_prob = (assigned_prob / np.sum(assigned_prob, axis=0).clip(0.01))


# Input the current state, prices, and demands
# Return a vector containing probability of transitioning to each other states 
def get_transition_prob_vector(current_state, prices, demands, transition_method = "uniform"):
    saved_budget = current_state.get_saved(prices, demands)
    # Return a uniform probability distribution for each state
    if transition_method == "uniform":
        transition_prob = np.ones(num_states) * 1/num_states
    # Return the assigned probability distribution
    elif transition_method == "assigned":
        transition_prob = assigned_prob[current_state.index]
    # Do something to adjust the transition probability
    elif transition_method == "adjusted":
        transition_prob = np.ones(num_states) * 1/num_states


# Input the transition probability vector of a given state
# Return the probability of transitioning to a certain state
def get_transition_prob(transition_prob_vector, next_state):
    return transition_prob_vector[next_state.index]



def value_iteration():
    pass




