import stochasticAlg as alg
import stochsticLibrary as lib
import numpy as np
import matplotlib.pyplot as plt





if __name__ == '__main__':
    num_iterations = 20
    max_iter_prices = 200
    max_iter_demands = 100
    learning_rates = [0.07,0.01,0.01] 
    num_experiment = 100
    num_buyers =  5
    num_goods = 8 
    num_states = 5
    initial_distribution = lib.normalize_vector(np.random.rand(num_states))
    discount = 0.5
    epsilon = 1e-400

    states = alg.create_states(num_buyers, num_goods, num_states, "linear",
    10, 10,             # valuations
    100, 0,             # budgets
    10, 10)             # supplies          


    v_history, exp_return_history = alg.value_iteration_dependent(discount, states, initial_distribution, num_iterations, epsilon, "linear", max_iter_prices, max_iter_demands, learning_rates)

    print(exp_return_history)

    plt.plot(exp_return_history)
    plt.ylabel('expected return')
    plt.show()
