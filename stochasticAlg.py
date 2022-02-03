import stochsticLibrary as lib
import numpy as np
from state import State
import fisherSolver as solver

num_states = 100
# Assigned probbaility of transition between states
assigned_prob = np.random.rand(num_states, num_states)




def create_states(num_buyers, num_goods, num_states, utility_type, valuations_range, valuations_add, budgets_range, budgets_add, supplies_range, supplies_add):
    states = []
    for  i in range(num_states):
        valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_add)
        budgets = lib.get_budgets(num_buyers, budgets_range, budgets_add)
        supplies = lib.get_supplies(num_goods, supplies_range, supplies_add)
        new_state = State(i, num_buyers, num_goods, valuations, budgets, supplies, utility_type)
        states.append(new_state)
    return states


# Input the current state, prices, and demands
# Return a vector containing probability of transitioning to each other states 
def get_transition_prob_vector(current_state, prices, demands, states, transition_method = "uniform"):
    num_states = len(states)
    saved_budget = current_state.get_saved(prices, demands)
    transition_prob = [0 for i in range(num_states)]
    
    # Return a uniform probability distribution for each state
    if transition_method == "uniform":
        transition_prob = np.ones(num_states) * 1/num_states
    # Return the assigned probability distribution
    elif transition_method == "randomly_assigned":
        normalized_assigned_prob = np.array([row/np.sum(row) for row in assigned_prob])
        transition_prob = normalized_assigned_prob[current_state.index]
    # Do something to adjust the transition probability
    elif transition_method == "close_to_saved_budget":
        for next_state in states:
            diff = np.linalg.norm(saved_budget - next_state.budgets)**2   # Difference between budgets in next state and saved budgets
            transition_prob.append(diff) # add different version!!!
        transition_prob = lib.normalize_vector(transition_prob)
        transition_prob = [1-prob for prob in transition_prob]
    elif transition_method == "close_to_demand" :
        for next_state in states:
            demands_sum = np.sum(demands, axis = 0)
            diff = np.linalg.norm(next_state.supplies - demands_sum)**2   # Difference between budgets in next state and saved budgets
            transition_prob.append(diff)
        transition_prob = lib.normalize_vector(transition_prob)
        transition_prob = [1-prob for prob in transition_prob]

    return transition_prob

def get_transition_prob(next_state_index, current_state, prices, demands, states, transition_method):
    return get_transition_prob_vector(current_state, prices, demands, states, transition_method)[next_state_index]



############# Projection onto affine positive half-space, i.e., budget set ###############
def project_to_bugdet_set(X, p, b):
    X_prec = X
    while (True): 
        X -= ((X @ p - b).clip(min= 0)/(np.linalg.norm(p)**2).clip(min= 0.01) * np.tile(p, reps = (b.shape[0], 1)).T).T
        X = X.clip(min = 0)
        # if(np.linalg.norm(X - X_prec) <= np.sum(X_prec)*0.05):
        if(np.linalg.norm(X - X_prec) <=0.0005):
            break
        # print(f"Current iterate {X}\nPrevious Iterate {X_prec}")
        X_prec = X
    return X
    


# The easy version (transition prabability is indepedent of actions)
def value_iteration_independent(discount, states, initial_distribution, num_iters, epsilon, utility_type):
    v_history = []
    exp_return_history = []

    # Initialze utility function
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        # util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        # util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        # util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj

    # Initialization
    num_states = len(states)
    v = np.zeros(num_states) # An vector that maps each state to its value v(s)
    num_buyers = states[0].num_buyers
    num_goods = states[0].num_goods
    prices_poliy = [np.zeros(num_goods) for i in range(num_states)]
    demands_policy = [np.zeros((num_buyers, num_states)) for i in range(num_states)]

    for i in range(num_iters):
        if (not i % 50):
            print(f" ----- Iteration {i}/{num_iters} ----- ")

        max_diff = 0 # Initialize max difference
        Tv = np.zeros(num_states)
        for state in states:
            index = state.index
            # Tv(s) = min max E[f(s,x,y) + discount * value(s')]
            #          = min max f(s,x,y) + discount * \sum_{s'} p(s'|s) * value(s')
            market = solver.FisherMarket(state.valuations, state.budgets, state.supplies)
            equil_demands, equil_prices = market.solveMarket(utility_type, printResults=False)
            equil_demands = np.array(equil_demands)
            obj = obj_func(equil_prices, equil_demands, state.supplies, state.budgets, state.valuations)
            prob_vector = get_transition_prob_vector(state, equil_prices, equil_demands, states, transition_method = "randomly_assigned")
            discounted_sum = discount * np.sum([prob_vector[i] * v[i] for i in range(num_states)])


            # Update policies and Tv
            prices_poliy[index] = equil_prices
            demands_policy[index] = equil_demands
            Tv[index] = obj + discounted_sum

            max_diff = max(max_diff, abs(v[index] - Tv[index]))

        # Update the values & expected returns
        v_history.append(v)
        exp_return_history.append(get_exp_return(v, initial_distribution))
        v = Tv
        # If diff smaller than threshold delta for all states, algorithm terminates
        if max_diff < epsilon:
            v_history.append(Tv)
            exp_return_history.append(get_exp_return(Tv, initial_distribution))
            break
    
    return v_history, exp_return_history





# The complicated version: close to saved budget
def value_iteration_dependent(discount, states, initial_distribution, max_iters, epsilon, utility_type, max_iter_prices, max_iter_demands, learning_rates):
    v_history = []
    exp_return_history = []
    num_buyers = states[0].num_buyers
    num_goods = states[0].num_goods

    # Initialze utility function
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        # util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        # util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        # util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj

    # Initialization
    num_states = len(states)
    v = np.zeros(num_states) # An vector that maps each state to its value v(s)
    prices_poliy = [np.zeros(num_goods) for index in range(num_states)]
    demands_policy = [np.zeros((num_buyers, num_goods)) for index in range(num_states)]

    for i in range(max_iters):
        print(f" ----- Value Iteration {i}/{max_iters} ----- ")
        max_diff = 0 # Initialize max difference
        Tv = np.zeros(num_states)
        for current_state in states:
            index = current_state.index
            # Tv(s) = min max E[f(s,x,y) + discount * value(s')]
            #          = min max f(s,x,y) + discount * \sum_{s'} p(s'| s, x, y) * value(s')

            # Update policies and Tv
            prices_poliy[index], demands_policy[index], Tv[index] = min_max_Q(num_buyers, num_goods, current_state, states, v, discount, max_iter_prices, max_iter_demands, learning_rates)
            max_diff = max(max_diff, abs(v[index] - Tv[index]))

        # Update the values & expected returns
        v_history.append(v)
        exp_return_history.append(get_exp_return(v, initial_distribution))
        v = (1 - discount) * Tv
        # If diff smaller than threshold delta for all states, algorithm terminates
        if max_diff < epsilon:
            v_history.append(Tv)
            exp_return_history.append(get_exp_return(Tv, initial_distribution))
            break
    
    return v_history, exp_return_history


def p_gradient_of_log(states, current_state, s_i, prices, demands):
    saved_budgets = current_state.get_saved(prices, demands)
    diff_sum = np.sum( [np.linalg.norm(next_state.budgets - saved_budgets)**2 for next_state in states] )
    diff_si = np.linalg.norm(s_i.budgets - saved_budgets)**2
    vector_part_sum = 2 * np.sum( [next_state.budgets - saved_budgets for next_state in states], axis = 0 )
    vector_part_si = 2 * (s_i.budgets - saved_budgets)
    return 1 / (diff_sum - diff_si) * ((vector_part_sum - vector_part_si) @ demands) - 1 / diff_sum * (vector_part_sum @ demands)

def x_i_gradient_of_log(row, states, current_state, s_i, prices, demands):
    num_buyers = current_state.num_buyers
    num_goods = current_state.num_goods
    saved_budgets = current_state.get_saved(prices, demands)
    diff_sum = np.sum( [np.linalg.norm(next_state.budgets - saved_budgets)**2 for next_state in states] )
    diff_si = np.linalg.norm(s_i.budgets - saved_budgets)**2
    vector_part_sum = 2 * np.sum( [next_state.budgets - saved_budgets for next_state in states], axis = 0 )
    vector_part_si = 2 * (s_i.budgets - saved_budgets)
    grad_of_Xp = np.zeros((num_buyers, num_goods))
    grad_of_Xp[row] = prices
    return 1 / (diff_sum - diff_si) * ((vector_part_sum - vector_part_si) @ grad_of_Xp) - 1 / diff_sum * (vector_part_sum @ grad_of_Xp)
    

# Compute min max Q = min max f(s,x,y) + discount * \sum_{s'} p(s'| s, x, y) * value(s')
def min_max_Q(num_buyers, num_goods, current_state, states, v, discount, max_iter_prices,  max_iter_demands, learning_rates):
    prices = np.random.rand(num_goods) * 10
    demands = np.random.rand(num_buyers, num_goods)
    lambdas = np.random.rand(num_buyers)
    num_states = len(states)

    for outer_iter in range(1, max_iter_prices):
        if (not outer_iter % 50):
            print(f"               ----- Min-Max Iteration {outer_iter}/{max_iter_prices} ----- ")
        ####### PRICES STEP ######
        # Calculate gradient of E[discount * v(s')] part
        sum = np.zeros(num_goods)
        for s_i in states:
            sum += v[s_i.index] * p_gradient_of_log(states, current_state, s_i, prices, demands)
        p_discounted_exp_of_v = discount * 1/num_states * sum
        # Calculate lagrangian part
        p_langrangian_part = np.sum( [lambdas[i] * demands[i] for i in range(num_buyers)] )
        # Update prices
        # prices = (prices - learning_rates[0] * (p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)
        prices = (prices - learning_rates[0] * outer_iter**(-1/2) * (p_discounted_exp_of_v - p_langrangian_part)).clip(min=0.001)


        for inner_iter in range(1, max_iter_demands):
            ###### DEMANDS STEP ######
            new_demands = np.zeros((num_buyers, num_goods))
            # Calculate obj parts
            constants_list = []
            for v_i, b_i, x_i in zip(current_state.valuations, current_state.budgets, demands):
                c_i = b_i / max(current_state.util_func(x_i, v_i), 0.001)
                constants_list.append(c_i)
            constants = (np.array(constants_list)).reshape(num_buyers, 1)
            obj_parts = constants *  current_state.util_gradient_func(demands, current_state.valuations)

            for row in range(num_buyers):
                # Calculate gradient of E[discount * v(s')] part
                sum = np.zeros(num_goods)
                for s_i in states:
                    sum += v[s_i.index] * x_i_gradient_of_log(row, states, current_state, s_i, prices, demands)
                X_discounted_exp_of_v = discount * 1/num_states * sum
                # Calculate langrangian part
                X_langrangian_part = lambdas[row] * prices
                # Update x_i
                demands[row] = (demands[row] + learning_rates[1] * inner_iter**(-1/2) * (obj_parts[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)
                # demands[row] = (demands[row] + learning_rates[1] * (obj_parts[row] + X_discounted_exp_of_v - X_langrangian_part)).clip(min=0.001)
                new_demands = project_to_bugdet_set(demands, prices, current_state.budgets)
                check_market(prices, demands, new_demands, current_state.budgets, current_state.supplies, current_state.valuations)
                demands = new_demands
            ###### LAMBDA STEP ######
            # lambdas = ( lambdas - learning_rates[2] * (current_state.budgets - prices@ demands.T )).clip(min=0.001)
            lambdas = ( lambdas - learning_rates[2] * inner_iter**(-1/2) * (current_state.budgets - prices@ demands.T )).clip(min=0.001)

    final_q = get_q_value(prices, demands, current_state, states, v, discount)
    return prices, demands, final_q




    





def simple_OFM(num_goods, num_buyers, valuations, budgets, distribution, num_iters, utility_type):

    # Initialze utility function
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        # util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        # util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        # util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj

    # Solve the market
    market = solver.FisherMarket(valuations, budgets, distribution)
    equil_demands, equil_prices = market.solveMarket(utility_type, printResults=False)
    equil_demands = np.array(equil_demands)
    
    sum_utility = np.zeros(num_buyers)
    average_utility = np.zeros(num_buyers)
    average_utility_history = []
    obj_history = []


    for t in range(1, num_iters):
        item = np.random.choice(range(num_goods), p = distribution)
        average_utility = np.zeros(num_buyers)
        # Keep track of the time-averaged utility
        for i, (v_i, x_i) in enumerate(zip(valuations, equil_demands)):
            utility = util_func(x_i / distribution[item], v_i)
            sum_utility[i] += utility
            average_utility[i] = sum_utility[i] / t
        average_utility_history.append(average_utility)

    return average_utility_history




########################## Evaluate the Algorithm #######################################3333
# Input the value function and the initial distribution
# Output the expected retrun
def get_exp_return(value, initial_distribution):
    num_states = len(value)
    return np.sum([initial_distribution[index] * value[index] for index in range(num_states)])


def get_q_value(prices, demands, current_state, states, v, discount):
    num_states = len(states)
    obj_part = current_state.obj_func(prices, demands, current_state.supplies, current_state.budgets, current_state.valuations)
    transition_prob_vector = get_transition_prob_vector(current_state, prices, demands, states, "close_to_saved_budget")
    expected_value_part = discount * np.sum( [transition_prob_vector[i] * v[i] for i in range(num_states)] )
    return obj_part + expected_value_part


def check_market(prices, demands, new_demands, budgets, supplies, valuations):
    remain_budgets = budgets - prices @ new_demands.T
    for remain_budget in remain_budgets:
        if remain_budget < -1:
            print("spendings exceeds budgets by", 0 - remain_budget)
    remain_supplies = supplies - np.sum(new_demands, axis = 0)
    # for remain_supply in remain_supplies:
    #     if remain_supply < -1:
    #         print("demands exceed supplies by", 0-remain_supply)
