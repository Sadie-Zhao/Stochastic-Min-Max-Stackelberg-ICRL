import stochsticLibrary as lib
import numpy as np
from state import State

num_states = 3
assigned_prob = np.random.rand(num_states, num_states)
assigned_prob = (assigned_prob / np.sum(assigned_prob, axis=0).clip(0.01))

assigned_prob = (assigned_prob / np.sum(assigned_prob, axis=0).clip(0.01))

print(assigned_prob)