import numpy as np
import stochsticLibrary as lib
class State:
    def __init__(self, index, num_goods, num_buyers, valuations, budgets, supplies, utility_type):
        self.index = index
        self.num_goods = num_goods
        self.num_buyers = num_goods
        self.num_buyers = num_buyers
        self.valuations = valuations
        self.budgets = budgets
        self.supplies = supplies
        self.utility_type = utility_type
        if utility_type == "linear":
            # self.util_func = lib.get_linear_utility
            self.reward_func = lib.get_linear_obj
        elif utility_type == "leontief":
            # self.util_func = lib.get_leontief_utility
            self.reward_func = lib.get_leontief_obj
        elif  utility_type == "cd":
            # self.util_func = lib.get_cd_utility
            self.reward_func = lib.get_cd_obj

    # Input the prices and demands
    # Return reward gained in this state
    def get_rewards(self, prices, demands):
        return self.reward_func(prices, demands, self.supplies, self.budgets, self.valuations)

    # Input the prices and demands
    # Return the money that buyers saved in this state when choosing certian prices and demands
    def get_saved(self, prices, demands):
        spending = prices @ demands.T
        return self.budgets - spending


