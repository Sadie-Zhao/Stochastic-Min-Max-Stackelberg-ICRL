import cvxpy as cp
import numpy as np
import sys

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class FisherMarket:
    """
    This is a class that models a Fisher Market

    Attributes:
        valutations (numpy: double x double):
            matrix of valuations of buyer i for good j.
        budgets (numpy: double): vector of buyer budgets.
        numGoods (numpy: double): vector of supplies for each good.

    """

    def __init__(self, V, B, S = None):
        """
        The constructor for the fisherMarket class.

        Parameters:
        valuations (numpy: double x double): matrix of valuations of buyer i for good j.
        budgets (numpy: double): vector of buyer budgets.
        numGoods (numpy: double): vector of supplies for each good.
        """
        self.budgets = B
        self.valuations = V
        self.numGoods = S

        if S is None:
            self.numGoods = np.ones(V.shape[1])

    def getBudgets(self):
        """
        Returns:
        The budgets of the buyers in the market
        """
        return self.budgets

    def getValuations(self):
        """
        Returns:
        The valuation of buyers in the market
        """
        return self.valuations

    def numberOfGoods(self):
        """
        Returns:
        Number of goods in the market
        """
        return self.numGoods.size

    def numberOfBuyers(self):
        """
        Returns:
        Number of buyers in the market
        """
        return self.budgets.size

    def getSupply(self):
        """
        Returns:
        Supply of each good
        """
        return self.numGoods

    def getCache(self):
        """
        Returns:
        The cached alocation values (X, p)
        """
        return (self.optimalX, self.optimalp)

    def solveMarket(self, utilities = "linear", printResults = True, rho = 1):
        """
        Parameters:
        utilities(string): Denotes the utilities used to solve market
            Currently options are 'quasilinear' and 'linear'

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """
        if (utilities == "quasilinear"):
            alloc, prices = self.solveQuasiLinear(printResults)
        elif (utilities == "linear"):
            alloc, prices = self.solveLinear(printResults)
        elif (utilities == "leontief"):
            alloc, prices = self.solveLeontief(printResults)
        elif (utilities == "cd"):
            alloc, prices = self.solveCobbDouglas(printResults)
        elif (utilities == "ces"):
            alloc, prices = self.solveCES(rho, printResults)
        else:
            print(f"Invalid Utility Model : '{utilities}'")
            sys.exit()

        supply = self.getSupply()

        p = prices

        X = alloc

        demand = np.sum(X, axis = 0)

        ##### Sanity Check ##### 

        # Check that the market clears
        assert (demand < supply*1.001).all(), f"Supply of Goods: {supply}\nDemand for Goods\
             Allocated: {demand}"


        return (X,p)


    def solveLinear(self, printResults = True):
        """
        Solves Fisher Market with Linear utilities

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """

        numberOfGoods = self.numberOfGoods()
        numberOfBuyers = self.numberOfBuyers()

        ########### Primal: Output => Allocation #########

        # Variables of program
        alloc = cp.Variable((numberOfBuyers, numberOfGoods))
        utils = cp.Variable(numberOfBuyers)

        # Objective
        obj = cp.Maximize(self.budgets.T @ cp.log(utils))

        constraints = [utils <= cp.sum(cp.multiply(self.valuations, alloc), axis = 1),
                        cp.sum(alloc, axis = 0) <= self.numGoods,
                        alloc >= 0]
        # Convex Program for primal
        primal = cp.Problem(obj, constraints)

        # Solve Program
        # objectiveValue = primal.solve()  # Returns the optimal value.
        try:
            objectiveValue = primal.solve(solver="ECOS")
        except cp.SolverError:
            objectiveValue = primal.solve(solver="SCS")

        X = alloc.value

        # Get dual variables, i.e., prices, directly from CVXPY
        p = constraints[1].dual_value
        
        if(printResults):
            print("Linear results:")
            print("Primal Status (Allocations): ", primal.status)
            print("Value of Primal Objective:, ", objectiveValue)
            print(f"Allocation: {X}\nPrices: {p}")

        return (X, p)


    def solveQuasiLinear(self, printResults = True):
        """
        Solves Fisher Market with quasilinear utilities

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """
        numberOfGoods = self.numberOfGoods()
        numberOfBuyers = self.numberOfBuyers()

        ########### Primal: Output => Allocation #########

        # Variables of program
        allocs = cp.Variable((numberOfBuyers, numberOfGoods))
        values = cp.Variable(numberOfBuyers)
        utils = cp.Variable(numberOfBuyers)

        # Objective
        obj = cp.Maximize(self.budgets.T @ cp.log(utils) - cp.sum(values))
        constraints = [utils <= (cp.sum(cp.multiply(self.valuations, allocs), axis = 1) + values),
                        cp.sum(allocs, axis = 0) <= self.numGoods,
                        allocs >= 0,
                        values >= 0]


        # Convex Program for dual, this analogous to the Eisenberg-Gale primal
        # The confusion is due to the inconsistency within the literature.
        
        dual = cp.Problem(obj, constraints)

        # Solve Program
        objectiveValue = dual.solve()  # Returns the optimal value.

        # Get primal maximizer (allocations)
        X = allocs.value

        # Get primal maximizer, i.e., prices, directly from CVXPY
        p = constraints[1].dual_value
        
        
        if(printResults):
            print("Linear results:")
            print("Status of program:", dual.status)
            print("Value of Dual Objective: ", objectiveValue)
            print(f"Allocation = {X}\nPrices = {p}")

        return (X, p)

    def solveLeontief(self, printResults):
        """
        Solves Fisher Market with Leontief utilities

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """
        numberOfGoods = self.numberOfGoods()
        numberOfBuyers = self.numberOfBuyers()

        ########### Primal: Output => Allocation #########

        # Variables of program
        alloc = cp.Variable((numberOfBuyers, numberOfGoods))
        utils = cp.Variable(numberOfBuyers)

        # Objective
        obj = cp.Maximize(self.budgets.T @ cp.log(cp.min(alloc / self.valuations, axis = 1)))
        # utilConstraints = [(utils[buyer] <= ( alloc[buyer,good] / self.valuations[numberOfBuyers - buyer -1,good])) for good in range(numberOfGoods) for buyer in range(numberOfBuyers)]
        constraints = [ cp.sum(alloc, axis = 0) <= self.numGoods,
                        alloc >= 0]
        constraints =  constraints

        # Convex Program for dual
        primal = cp.Problem(obj, constraints)

        # Solve Program
        # objectiveValue = primal.solve()  # Returns the optimal value.
        try:
            objectiveValue = primal.solve(solver="ECOS")
        except cp.SolverError:
            objectiveValue = primal.solve(solver="SCS")


        # Get primal maximizer
        X = alloc.value
        # Get dual value (for prices) directly from CVXPY
        p = constraints[0].dual_value
        
        #### Print Option ####
        if(printResults):
            print("Leontief results:")
            print("Status of program:", primal.status)
            print("Value of Primal Objective: ", objectiveValue)
            print(f"Allocation = {X}\nPrices = {p}")
    
        return (X,p)

    def solveCobbDouglas(self, printResults):
        """
        Solves Fisher Market with Cobb-Douglas utilities

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """

        valuations = self.getValuations()
        budgets = self.getBudgets()
        
        # Equilibrium for Cobb-Douglas Fisher market is given in closed form
        normalized_vals = (valuations.T /np.sum(valuations, axis = 1)).T 

        p = normalized_vals.T @ budgets
        X = (normalized_vals.T * budgets).T / p

        if(printResults):
            print("CD results:")
            print(f"Allocation: {X}\nPrices: {p}")
        
        return (X, p)


    def solveCES(self, rho, printResults = True):
        """
        Solves Fisher Market with Linear utilities

        Returns:
        A tuple (X, p) that corresponds to the optimal matrix of allocations and
        prices.
        """

        numberOfGoods = self.numberOfGoods()
        numberOfBuyers = self.numberOfBuyers()

        ########### Primal: Output => Allocation #########

        # Variables of program
        alloc = cp.Variable((numberOfBuyers, numberOfGoods))
    
        # Objective
        obj = cp.Maximize(self.budgets.T @ ((1/rho)*cp.log(cp.sum(cp.multiply(self.valuations,cp.power(alloc, rho)), axis = 1))))

        constraints = [ cp.sum(alloc, axis = 0) <= 1,
                        alloc >= 0]

        # Convex Program for primal
        primal = cp.Problem(obj, constraints)


        # Solve Program
        objectiveValue = primal.solve()  # Returns the optimal value.

        X = alloc.value

        # Get dual minimizer, i.e., prices, directly from CVXPY
        p = constraints[0].dual_value

        if(printResults):
            print("Primal Status (Allocation):", primal.status)
            print("Value of Primal Objective: ", objectiveValue)
            print(f"Allocation: {X}\nPrices: {p}")

        return (X, p)