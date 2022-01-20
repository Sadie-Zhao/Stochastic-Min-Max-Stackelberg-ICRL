import stochasticAlg as alg
import stochsticLibrary as lib
import numpy as np
import matplotlib.pyplot as plt

num_goods = 20
num_buyers = 5
valuations = lib.get_valuations(num_buyers, num_goods, 10,10)
budgets = lib.get_budgets(num_buyers, 100, 0)
distribution = lib.normalize_vector(np.random.rand(num_goods))
num_iters = 10000
utility_type = "linear"

average_utility_history = alg.simple_OFM(num_goods, num_buyers, valuations, budgets, distribution, num_iters, utility_type)

avg_util_0 = [avg_util[4] for avg_util in average_utility_history]

print(avg_util_0)
plt.plot(avg_util_0)
plt.ylabel('avg_util for buyer 0')
plt.show()