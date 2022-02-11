import stochasticAlg as alg
import stochsticLibrary as lib
import numpy as np
import matplotlib.pyplot as plt






if __name__ == '__main__':
    num_iterations = 50
    max_iter_prices = 100
    max_iter_demands = 100
    learning_rates = [0.07,0.01,0.001] 
    learning_rates_1 = [0.5,0.01]
    num_experiment = 100
    num_goods = 3
    num_buyers = 2
    num_states = 5
    initial_distribution = lib.normalize_vector(np.random.rand(num_states))
    discount = 0.75
    epsilon = 1e-400

    states = alg.create_states(num_buyers, num_goods, num_states, "linear",
    10, 10,             # valuations
    100, 0,             # budgets
    10, 10)             # supplies    

    states[0].budgets = np.ones(num_buyers)* 0.0001 


    v_history, exp_return_history, total_excess_demands, total_excess_spendings, prices_policy, demands_policy = alg.value_iteration_saving(discount, states, 
    initial_distribution, num_iterations, epsilon, "linear", max_iter_prices, max_iter_demands, learning_rates_1)


    plt.plot(exp_return_history)
    plt.ylabel('expected return')
    plt.show()
    plt.savefig("test.png")
    # plt.savefig('value_iteration_{}_{}_{}_{}.png'.format(num_iterations, max_iter_prices, max_iter_demands, discount))

    plt.plot(total_excess_demands)
    plt.ylabel('excess_demands')
    plt.show()
    # plt.savefig('excess_demands_{}_{}_{}_{}.png'.format(num_iterations, max_iter_prices, max_iter_demands, discount))

    plt.plot(total_excess_spendings)
    plt.ylabel('excess_spendings')
    plt.show()
    # plt.savefig('excess_spendings_{}_{}_{}_{}.png'.format(num_iterations, max_iter_prices, max_iter_demands, discount))

    file_name = 'resluts.npy'
    with open(file_name, 'wb') as f:
        np.save(f, exp_return_history)
        np.save(f, total_excess_demands)
        np.save(f, total_excess_spendings)
        np.save(f, prices_policy)
        np.save(f, demands_policy)


    with open(file_name, 'rb') as f:
        exp_return_history = np.load(f)
        total_excess_demands = np.load(f)
        total_excess_spendings = np.load(f)
        prices_policy = np.load(f)
        demands_policy = np.load(f)

