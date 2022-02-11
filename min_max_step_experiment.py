
from concurrent.futures import process
import stochasticAlg as alg
import stochsticLibrary as lib
import numpy as np
from state import State
import fisherSolver as solver
import matplotlib.pyplot as plt
import math


# # Compute min max Q = min max f(s,x,y) + discount * \sum_{s'} p(s'| s, x, y) * value(s')
# def min_max_Q_test(num_buyers, num_goods, current_state, states, v, discount, max_iter_prices,  max_iter_demands, learning_rates):
#     prices_step_sizes = []
#     demands_step_sizes = []
#     lambda_step_sizes = []
#     total_excess_spendings = []
#     total_excess_demands = []
#     prices = np.random.rand(num_goods) * 10
#     demands = np.random.rand(num_buyers, num_goods)
#     lambdas = np.random.rand(num_buyers)
#     num_states = len(states)

#     for outer_iter in range(1, max_iter_prices):
#         if (not outer_iter % 50):
#             print(f"               ----- Min-Max Iteration {outer_iter}/{max_iter_prices} ----- ")
#         ####### PRICES STEP ######
#         demand_row_sum = np.sum(demands, axis = 0)
#         excess_demands = demand_row_sum - current_state.supplies

#         # prices_step_size = learning_rates[0] * outer_iter**(-1/2) * excess_demands
#         prices_step_size = learning_rates[0] * excess_demands

#         prices += prices_step_size * (prices > 0)
#         prices_step_sizes.append(np.linalg.norm(prices_step_size))
#         prices = np.clip(prices, a_min=0.001, a_max = None) # Make sure the price is positive
#         # step_size = learning_rates[0] * (iter**(-1/2)) * excess_demands

#         # # Calculate obj parts
#         # prices_obj = current_state.supplies
#         # # Calculate gradient of E[discount * v(s')] part
#         # sum = np.zeros(num_goods)
#         # for s_i in states:
#         #     sum += v[s_i.index] * alg.p_gradient_of_log(states, current_state, s_i, prices, demands)
#         # p_discounted_exp_of_v = discount * 1/num_states * sum
#         # # Calculate lagrangian part
#         # p_langrangian_part = np.sum( [lambdas[i] * demands[i] for i in range(num_buyers)] )
#         # # Update prices
#         # # prices = (prices - learning_rates[0] * (p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)
#         # prices = (prices - learning_rates[0] * outer_iter**(-1/2) * (prices_obj + p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)



#         for inner_iter in range(1, max_iter_demands):
#             ###### DEMANDS STEP ######
#             new_demands = np.zeros((num_buyers, num_goods))
#             # Calculate obj parts
#             constants_list = []
#             for v_i, b_i, x_i in zip(current_state.valuations, current_state.budgets, demands):
#                 c_i = b_i / max(current_state.util_func(x_i, v_i), 0.001)
#                 constants_list.append(c_i)
#             constants = (np.array(constants_list)).reshape(num_buyers, 1)
#             demands_obj = constants *  current_state.util_gradient_func(demands, current_state.valuations)

#             for row in range(num_buyers):
#                 # Calculate gradient of E[discount * v(s')] part
#                 sum = np.zeros(num_goods)
#                 for s_i in states:
#                     sum += v[s_i.index] * alg.x_i_gradient_of_log(row, states, current_state, s_i, prices, demands)
#                 X_discounted_exp_of_v = discount * 1/num_states * sum
#                 # Calculate langrangian part
#                 X_langrangian_part = lambdas[row] * prices
#                 # Update x_i
#                 # new_demands[row] = (demands[row] + learning_rates[1] * inner_iter**(-1/2) * (demands_obj[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)
#                 new_demands[row] = (demands[row] + learning_rates[1] * (demands_obj[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)

#             # new_demands = alg.project_to_bugdet_set(demands, prices, current_state.budgets)
#             # alg.check_market(prices, new_demands, current_state.budgets, current_state.supplies, total_excess_spendings, total_excess_demands)
#             demands_step_sizes.append(np.linalg.norm(new_demands-demands))
#             demands = new_demands


#             ##### LAMBDA STEP ######
#             lambda_step_size = learning_rates[2] * (current_state.budgets - prices@ demands.T )
#             lambdas = ( lambdas - lambda_step_size).clip(min=0.001)
#             # lambdas = ( lambdas - learning_rates[2] * inner_iter**(-1/2) * lambda_step_size).clip(min=0.001)
#             lambda_step_sizes.append(np.linalg.norm(lambda_step_size))
        
#         alg.check_market(prices, new_demands, current_state.budgets, current_state.supplies, total_excess_spendings, total_excess_demands)
#     # final_q = alg.get_q_value(prices, demands, current_state, states, v, discount)
#     excess_spendings =  prices @ demands.T - current_state.budgets
#     excess_demands = np.sum(demands, axis =0) - current_state.supplies
#     print(excess_spendings, excess_demands)
#     return prices_step_sizes, demands_step_sizes, lambda_step_sizes, total_excess_spendings, total_excess_demands


# Compute min max Q = min max f(s,x,y) + discount * \sum_{s'} p(s'| s, x, y) * value(s')
def min_max_Q_new(num_buyers, num_goods, current_state, states, v, discount, max_iter_prices, learning_rates):
    prices_step_sizes = []
    demands_step_sizes = []
    total_excess_spendings = []
    total_excess_demands = []
    prices = np.random.rand(num_goods) * 10 + 100
    demands = np.random.rand(num_buyers, num_goods)
    num_states = len(states)

    for outer_iter in range(1, max_iter_prices):
        if (not outer_iter % 50):
            print(f"               ----- Min-Max Iteration {outer_iter}/{max_iter_prices} ----- ")
        ####### PRICES STEP ######
        demand_row_sum = np.sum(demands, axis = 0)
        excess_demands = demand_row_sum - current_state.supplies

        new_prices = prices + learning_rates[0] * (outer_iter)**(-1/2) * excess_demands * (prices > 0)
        # new_prices = prices + learning_rates[0] * excess_demands * (prices > 0)
        new_prices = np.clip(new_prices, a_min=0.001, a_max = None)
        prices_step_sizes.append(np.linalg.norm(new_prices - prices))
        prices = new_prices
        # step_size = learning_rates[0] * (iter**(-1/2)) * excess_demands

        # # Calculate obj parts
        # prices_obj = current_state.supplies
        # # Calculate gradient of E[discount * v(s')] part
        # sum = np.zeros(num_goods)
        # for s_i in states:
        #     sum += v[s_i.index] * alg.p_gradient_of_log(states, current_state, s_i, prices, demands)
        # p_discounted_exp_of_v = discount * 1/num_states * sum
        # # Calculate lagrangian part
        # p_langrangian_part = np.sum( [lambdas[i] * demands[i] for i in range(num_buyers)] )
        # # Update prices
        # # prices = (prices - learning_rates[0] * (p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)
        # prices = (prices - learning_rates[0] * outer_iter**(-1/2) * (prices_obj + p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)



        ###### DEMANDS STEP ######
        new_demands = np.zeros((num_buyers, num_goods))
        # Calculate obj parts
        constants_list = []
        for v_i, b_i, x_i in zip(current_state.valuations, current_state.budgets, demands):
            c_i = b_i / max(current_state.util_func(x_i, v_i), 0.001)
            constants_list.append(c_i)
        constants = (np.array(constants_list)).reshape(num_buyers, 1)
        demands_obj = constants *  current_state.util_gradient_func(demands, current_state.valuations)

        for row in range(num_buyers):
            # Calculate gradient of E[discount * v(s')] part
            sum = np.zeros(num_goods)
            for s_i in states:
                sum += v[s_i.index] * alg.x_i_gradient_of_log(row, states, current_state, s_i, prices, demands)
            X_discounted_exp_of_v = discount * 1/num_states * sum
            # Calculate langrangian part
            X_langrangian_part = prices
            # Update x_i
            new_demands[row] = (demands[row] + learning_rates[1] * outer_iter**(-1/2) * (demands_obj[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)
            # new_demands[row] = (demands[row] + learning_rates[1] * (demands_obj[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)

        # new_demands = alg.project_to_bugdet_set(demands, prices, current_state.budgets)
        alg.check_market(prices, new_demands, current_state.budgets, current_state.supplies, total_excess_spendings, total_excess_demands)
        demands_step_sizes.append(np.linalg.norm(new_demands-demands))
        demands = new_demands
        print(prices, demands)
        # print(prices)
        
        # alg.check_market(prices, new_demands, current_state.budgets, current_state.supplies, total_excess_spendings, total_excess_demands)
    # final_q = alg.get_q_value(prices, demands, current_state, states, v, discount)
    return prices_step_sizes, demands_step_sizes, total_excess_spendings, total_excess_demands




num_goods = 3
num_buyers = 2
num_states = 5
discount = 0.75
states = alg.create_states(num_buyers, num_goods, num_states, "linear", 10, 10, 100, 0, 10, 10)
# Enforce one state to have zero budgets

states[0].budgets = np.ones(num_buyers)* 0.0001 
current_state = states[0]
s_i = states[1]
# prices = np.random.rand(num_goods) + 10000
# demands = np.random.rand(num_buyers, num_goods)
# lambdas = np.random.rand(num_buyers) + 10000


learning_rates = [5, 0.01]   ##### Fix 0.01 and 0.001 for demands and lambdas!!
learning_rates_1 = [0.5,0.001]

v = np.random.rand(num_states) * 200 + 5

prices_step_sizes, demands_step_sizes, total_excess_spendings, total_excess_demands= min_max_Q_new(num_buyers, num_goods, 
current_state, states, v, discount, 500, learning_rates_1)

# print(prices_step_sizes)
plt.plot(prices_step_sizes)
plt.ylabel("prices step sizes")
plt.show()

# print(demands_step_sizes)
plt.plot(demands_step_sizes)
plt.ylabel("demands step sizes")
plt.show()

# # print(lambda_step_sizes)
# plt.plot(lambda_step_sizes)
# plt.ylabel("lambda step sizes")
# plt.show()



plt.plot(total_excess_spendings)
plt.ylabel("excess spendings")
plt.show()

plt.plot(total_excess_demands)
plt.ylabel("excess demands")
plt.show()

