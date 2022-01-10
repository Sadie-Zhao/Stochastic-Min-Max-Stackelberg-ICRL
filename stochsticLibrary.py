# Library Imports
from cvxpy import constraints
import consumerUtility as cu
import numpy as np
import cvxpy as cp
import fisherSolver as solver
from numpy.core.numeric import ones
from numpy.lib.twodim_base import triu_indices_from
np.set_printoptions(precision=3)


#################### Market Parameters #######################
def get_valuations(num_buyers, num_goods, range, addition_term):
    valuations = np.random.rand(num_buyers, num_goods) * range + addition_term
    return valuations

def get_budgets(num_buyers, range, addition_term):
    budgets = np.random.rand(num_buyers) * range + addition_term
    return budgets

def get_supplies(num_goods, range, addition_term):
    supplies = np.random.rand(num_goods) * range + addition_term
    return supplies



##################### Utility Function ########################

# Linear utility function: Perfect Substitutes
def get_linear_utility(allocation, valuations): 
    # Returns utility value for linear agent
    return allocation.T @ valuations

# Leontief utility function: Perfect Complements
def get_leontief_utility(allocation, valuations):
    return np.min(allocation / valuations)

# Cobb-Douglas utility function
def get_cd_utility(allocation, valuations):
    # For easy of calculation, normalize valautions
    # This does not change the preference relation that the utility function represents
    normalized_vals = valuations / np.sum(valuations)
    return np.prod(np.power(allocation, normalized_vals))



#################### Utility Gradient Functions #####################
def get_linear_util_gradient(allocations, valuations):
    return valuations

def get_leontief_util_gradient(allocations, valuations):
    grad_matrix = []
    for x_i, v_i in zip(allocations, valuations):
        argmin_j = np.argmin([v_ij / x_ij for v_ij, x_ij in zip(x_i, v_i)])
        grad_array = np.zeros(len(x_i))
        grad_array[argmin_j] = 1 / v_i[argmin_j]
        grad_matrix.append(grad_array)
    
    return np.array(grad_matrix)

def get_cd_util_gradient(allocations, valuations):
    return ( np.prod(np.power(allocations, valuations), axis = 1)*( valuations / allocations.clip(min = 0.001) ).T).T




################### Objective Functions ##########################

def get_linear_obj(prices, demands, supplies, budgets, valuations):
    utils = np.sum(valuations * demands, axis = 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min=0.0001))


def get_leontief_obj(prices, demands, supplies, budgets, valuations):
    utils = np.min(demands/valuations, axis = 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min= 0.0001))


def get_cd_obj(prices, demands, supplies, budgets, valuations):
    utils = np.prod(np.power(demands, valuations), axis= 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min=0.0001))

def get_average_obj_static(obj_func, prices_hist, demands_hist, budgets, valuations):
    obj_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        obj = obj_func(p, x, budgets, valuations)
        obj_avg_hist.append(obj)
    return obj_avg_hist

def get_average_obj_dynamic(obj_func, prices_hist, demands_hist, budgets_hist, valuations_hist):
    obj_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        budgets = np.mean(np.array(budgets_hist[:i+1]).clip(min = 0), axis = 0)
        valuations = np.mean(np.array(valuations_hist[:i+1]).clip(min = 0), axis = 0)
        obj = obj_func(p, x, budgets, valuations)
        obj_avg_hist.append(obj)
    return obj_avg_hist






#################### Value Functions ###################################
def get_linear_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
    # return np.sum(prices) + budgets.T @ np.log(utils) + np.sum(budgets) - np.sum(demands @ prices )
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.01))

def get_cd_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_CD_indirect_util(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 

def get_leontief_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 


def get_average_value_static(value_func, prices_hist, demands_hist, budgets, valuations):
    value_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        value = value_func(p, x, budgets, valuations)
        value_avg_hist.append(value)
    return value_avg_hist

def get_average_value_dynamic(value_func, prices_hist, demands_hist, budgets_hist, valuations_hist):
    value_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        budgets = np.mean(np.array(budgets_hist[:i+1]).clip(min = 0), axis = 0)
        valuations = np.mean(np.array(valuations_hist[:i+1]).clip(min = 0), axis = 0)
        value = value_func(p, x, budgets, valuations)
        value_avg_hist.append(value)
    return value_avg_hist