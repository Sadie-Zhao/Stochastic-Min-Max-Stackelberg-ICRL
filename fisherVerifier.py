import cvxpy as cp
import numpy as np


def verify(X, p, V, b, utility, M = None, rho = None):
    """
    Given and allocation of goods, prices, valuations and utility types, check if
    a pair of allocation and prices satisfy conditions for a competitive
    equilibrium, that is 1) supply_j * p_j >= demand_j * p_j and 2) the
    allocation is envy free.
    Inputs:
    X : A matrix of allocations (size: |buyers| x |goods|)
    p : A vector of prices for goods (size: |goods|)
    V : A matrix of valuations (size: |buyers| x |goods| )
    b : A vector of budgets (size: |buyers|)
    utility: The type of utilty function
    M : Number of goods
    Returns:
    Boolean: True if (X, p) form a CE for valuations V and budgets b,
    false otherwise
    """
    numberOfBuyers, numberOfGoods = V.shape

    alloc = np.zeros((numberOfBuyers, numberOfGoods))
    for i in range(numberOfBuyers):
        alloc[i,:] = getMarshallianDemand(V[i,:], p, b[i], utility, rho = rho)

    print("demands:", X)
    print("valuations:", V)
    print("budgets:", b)
    print("prices:", p)
        
    print(f"Input Allocation:\n{X}\nVerifier Allocation:\n{alloc}")

def getMarshallianDemand(valuation, prices, budget, utility = "linear", rho = None):
    """
    Given a vector of consumer valuations, v, a price vector, p, and a budget, compute the utility
    of a utility-maximizing bundle. Mathematically, solve the linear program:
    max_{x} xv
    s.t. xp <= budget
    :param valuation: a consumer's valuation for goods.
    :param prices: prices of goods.
    :param budget: the consumer's budget
    :return:
    """
  
    num_items = len(valuation)
    x = cp.Variable(num_items)
    
    if (utility == "linear"):
        obj = cp.Maximize(x.T @ valuation)
    elif (utility == "leontief"):
        obj = cp.Maximize( cp.min( cp.multiply(x, 1/valuation) ) )
    elif (utility == "cobb-douglas"):
        obj = cp.Maximize( valuation.T @ cp.log(x))
    elif (utility == "ces"):
        x_rho = cp.power(x, rho)
        util = valuation.T @ x_rho
        obj = cp.Maximize((1/rho)*cp.log(util))
    else:
        obj = cp.Maximize(x.T @ (valuation - prices))

    constraints = [ (x.T @ prices) <= budget,
                    x >= 0]
    prob = cp.Problem(obj, constraints)

    prob.solve()

    return x.value

def getIndirectUtil(valuation, prices, budget, utility = "linear", rho = None):
    """
    Given a vector of consumer valuations, v, a price vector, p, and a budget, compute the utility
    of a utility-maximizing bundle. Mathematically, solve the linear program:
    max_{x} xv
    s.t. xp <= budget
    :param valuation: a consumer's valuation for goods.
    :param prices: prices of goods.
    :param budget: the consumer's budget
    :return:
    """
  
    num_items = len(valuation)
    x = cp.Variable(num_items)
    
    if (utility == "linear"):
        obj = cp.Maximize(x.T @ valuation)
    elif (utility == "leontief"):
        obj = cp.Maximize( cp.min( cp.multiply(x, 1/valuation) ) )
    elif (utility == "cobb-douglas"):
        obj = cp.Maximize( cp.sum(cp.multiply(valuation, cp.log(x))))
    elif (utility == "ces"):
        x_rho = cp.power(x, rho)
        util = valuation.T @ x_rho
        obj = cp.Maximize((1/rho)*cp.log(util))
    else:
        obj = cp.Maximize(x.T @ (valuation - prices))

    constraints = [ (x.T @ prices) <= budget,
                    x >= 0]
    prob = cp.Problem(obj, constraints)

    return prob.solve()