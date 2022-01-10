# Library Imports
import numpy as np
import cvxpy as cp
np.set_printoptions(precision=3)



############# Important Utility Function Definitions #############

# Linear utility function: Perfect Substitutes
def get_linear_utility(allocation, valuations):
    # Returns utility value for linear agent
    return allocation.T @ valuations

# Quasilinear utility function
def get_quasilinear_utility(allocation, valuations, prices):
    # Returns utility value for linear agent
    return allocation.T @ (valuations - prices)

# CES utility Function, - infinity < rho < 1
def get_ces_utility(allocation, valuations, rho):
    # Returns utility value for CES agent
    return np.power(np.power(allocation, rho).T @ valuations, (1/rho))

# Leontief utility function: Perfect Complements
def get_leontief_utility(allocation, valuations):
    return np.min(allocation/valuations)

# Cobb-Douglas utility function
def get_CD_utility(allocation, valuations):
    # For easy of calculation, normalize valautions
    # This does not change the preference relation that the utility function represents
    normalized_vals = valuations / np.sum(valuations)
    return np.prod(np.power(allocation, normalized_vals))
    
############# Economic Function Definitions #############

###### Define UMP and EMP functions for Custom utility functions ######


def get_custom_ump(prices, budget, util_func):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative scalar valued budget
        util_func: A utility function which takes as input a vector of allocation and ouputs utility
        Note that the utility function should already be parametrized when inputted into the function
    Returns:
        (indirect_utility, marshallian_demand): A UMP solution pair
            indirect_utility: The maximum utility achievable at inputted prices and budget
            marshallian_demand: The vector of goods that maximizes utility constrained by budget
    """

    allocation = cp.Variable(prices.shape[0])
    obj = cp.Maximize(util_func(allocation))
    constr = [allocation.T @ prices <= budget,
                allocation >= 0 ]
    program = cp.Problem(obj, constr)

    indirect_util = program.solve()
    marshallian_demand = allocation.value

    return (indirect_util, marshallian_demand)

def get_custom_indirect_util(prices, budget, util_func):
    return get_custom_ump(prices, budget, util_func)[0]

def get_custom_marshallian_demand(prices, budget, util_func):
    return get_custom_ump(prices, budget, util_func)[1]

def get_custom_emp(prices, util_level, util_func):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative utility level that the buyer wants to at least achieve
        util_func: A utility function which takes as input a vector of allocation and ouputs utility
        Note that the utility function should already be parametrized when passed into the function
    Returns:
        (expenditure, hicksian_demand): A EMP solution pair
            expenditure: The minimum spending needed to achieve a utility level of 
                util_level at inputted prices
            hicksian: The vector of goods that achieves util_level units of utility and 
                minimizes expenditure at inputted prices
    """
    allocation = cp.Variable(prices.shape[0])
    obj = cp.Minimize(prices.T @ allocation)
    constr = [util_func(allocation) >= util_level,
              allocation >= 0 ]
    program = cp.Problem(obj, constr)

    expenditure = program.solve()
    hicksian_demand = allocation.value

    return (expenditure, hicksian_demand)

def get_custom_expend(prices, util_level, util_func):
    # Returns the expenditure for linear utility
    return get_custom_emp(prices, util_level, util_func)[0]

def get_custom_hicksian_demand(prices, util_level, util_func):
    # Returns the hicksian demand for linear utilities
    return get_custom_emp(prices, util_level, util_func)[1]


###### Define UMP and EMP functions for Leontief, Cobb-Douglas and CES utility functions ######
# Recall that closed form solutions exist for the UMP and EMP problem for both of these function


######    Linear Utilities    ######

def get_linear_indirect_utill(prices, budget, valuations):
    return np.max(valuations/prices)*budget

def get_linear_marshallian_demand(prices, budget, valuations):
    max_bang_buck_goods = (valuations/prices >= np.max(valuations/prices))
    cost = max_bang_buck_goods.T @ prices
    
    return max_bang_buck_goods*(budget/cost)

def get_linear_expend(prices, util, valuations):
    return np.min(prices/valuations)*util

###### Cobb-Douglas Utilities ######


def get_CD_marshallian_demand(prices, budget, valuations):
    normalized_vals = valuations / np.sum(valuations)
    return normalized_vals*budget/prices.clip(min = 0.00001)

def get_CD_indirect_util(prices, budget, valuations):
    return get_CD_utility(get_CD_marshallian_demand(prices, budget, valuations), valuations)

def get_CD_ump(prices, budget, valuations):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative scalar valued budget
        valuations: A vector of valuations that parametrizes Cobb-Douglas utilities
    Returns:
        (indirect_utility, marshallian_demand): A UMP solution pair
            indirect_utility: The maximum utility achievable at inputted prices and budget
            marshallian_demand: The vector of goods that maximizes utility constrained by budget
    """
    indirect_util = get_CD_indirect_util(prices, budget, valuations)
    marshallian_demand = get_CD_marshallian_demand(prices, budget, valuations)

    return (indirect_util, marshallian_demand)

def get_CD_expend(prices, util, valuations):
    normalized_vals = valuations / np.sum(valuations)
    K = 1/np.prod((np.power(normalized_vals, normalized_vals)))
    return K*np.prod(np.power(prices, normalized_vals))*util

def get_CD_hicksian_demand(prices, util, valuations):
    return get_CD_marshallian_demand(prices, get_CD_expend(prices, util, valuations), valuations)

def get_CD_emp(prices, util_level, valuations):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative utility level that the buyer wants to at least achieve
        valuations: A vector of valuations that parametrizes the Cobb-Douglas utility
    Returns:
        (expenditure, hicksian_demand): A EMP solution pair
            expenditure: The minimum spending needed to achieve a utility level of 
                util_level at inputted prices
            hicksian: The vector of goods that achieves util_level units of utility and 
                minimizes expenditure at inputted prices
    """
    expenditure = get_CD_expend(prices, util_level, valuations)
    hicksian_demand = get_CD_hicksian_demand(prices, util_level, valuations)

    return (expenditure, hicksian_demand)

###### Leontief Utilities ######

def get_leontief_indirect_util(prices, budget, valuations):
    return budget/(prices.T @ valuations)

def get_leontief_marshallian_demand(prices, budget, valuations):
    return (budget/(prices.T @ valuations)) * valuations

def get_leontief_ump(prices, budget, valuations):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative scalar valued budget
        valuations: A vector of valuations that parametrizes the Leontief utility
    Returns:
        (indirect_utility, marshallian_demand): A UMP solution pair
            indirect_utility: The maximum utility achievable at inputted prices and budget
            marshallian_demand: The vector of goods that maximizes utility constrained by budget
    """
    indirect_util = get_leontief_indirect_util(prices, budget, valuations)
    marshallian_demand = get_leontief_marshallian_demand(prices, budget, valuations)

    return (indirect_util, marshallian_demand)

def get_leontief_expend(prices, util_level, valuations):
    return util_level*(prices.T @ valuations)

def get_leontief_hicksian_demand(prices, util_level, valuations):
    return valuations*util_level

def get_leontief_emp(prices, util_level, valuations):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative utility level that the buyer wants to at least achieve
        valuations: A vector of valuations that parametrizes the Leontief utility
    Returns:
        (expenditure, hicksian_demand): A EMP solution pair
            expenditure: The minimum spending needed to achieve a utility level of 
                util_level at inputted prices
            hicksian: The vector of goods that achieves util_level units of utility and 
                minimizes expenditure at inputted prices
    """
    expenditure = get_leontief_expend(prices, util_level, valuations)
    hicksian_demand = get_leontief_hicksian_demand(prices, util_level, valuations)

    return (expenditure, hicksian_demand)

###### CES Utilities ######

def get_ces_indirect_util(prices, budget, valuations, rho):
    sigma = 1/(1-rho)
    v = np.power(valuations, sigma)
    p = np.power(prices, 1-sigma)
    cost_unit_util = np.power(v.T @ p, 1/(1-sigma))
    return budget/cost_unit_util

def get_ces_marshallian_demand(prices, budget, valuations, rho):
    v = np.power(valuations, 1/(1-rho))
    p_num = np.power(prices, 1/(rho-1))
    p_denom = np.power(prices, rho/(rho-1))
    return budget* ((v*p_num)/ (v.T @ p_denom))

def get_ces_ump(prices, budget, valuations, rho):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative scalar valued budget
        valuations: A vector of valuations that parametrizes the CES utility
    Returns:
        (indirect_utility, marshallian_demand): A UMP solution pair
            indirect_utility: The maximum utility achievable at inputted prices and budget
            marshallian_demand: The vector of goods that maximizes utility constrained by budget
    """
    indirect_util = get_ces_indirect_util(prices, budget, valuations, rho)
    marshallian_demand = get_ces_marshallian_demand(prices, budget, valuations, rho)

    return (indirect_util, marshallian_demand)

def get_ces_expend(prices, util_level, valuations, rho):
    # Returns the expenditure for CES utility
    sigma = 1/(1-rho)
    v = np.power(valuations, sigma)
    p = np.power(prices, 1-sigma)
    cost_unit_util = np.power(v.T @ p, 1/(1-sigma))
    return util_level*cost_unit_util

def get_ces_hicksian_demand(prices, util_level, valuations ,rho):
    # Returns the hicksian demand for CES utilities
    expenditure = get_ces_expend(prices, util_level, valuations, rho)
    return get_ces_marshallian_demand(prices, expenditure, valuations, rho)
    
def get_ces_emp(prices, util_level, valuations, rho):
    """
    Inputs: 
        prices: A vector of non-negative prices
        budget: A non-negative utility level that the buyer wants to at least achieve
        valuations: A vector of valuations that parametrizes the CES utility
        rho: The elasticity of substitution of the utility function
    Returns:
        (expenditure, hicksian_demand): A EMP solution pair
            expenditure: The minimum spending needed to achieve a utility level of 
                util_level at inputted prices
            hicksian: The vector of goods that achieves util_level units of utility and 
                minimizes expenditure at inputted prices
    """
    expenditure = get_ces_expend(prices, util_level, valuations, rho)
    hicksian_demand = get_ces_hicksian_demand(prices, util_level, valuations, rho)

    return (expenditure, hicksian_demand)