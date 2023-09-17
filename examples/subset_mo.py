"""Find the optimal subset fulfilling any property. 
Multi-objective variant of https://github.com/dietmarwo/fast-cma-es/blob/master/examples/subset.py 

As example we use transactions where a subset is to be matched to a list of payments. 
For transactions: [2,4,5,1,3] and payments: [4, 4] the sum of payments is 8. 
Subsets of transactions that minimize the difference between payments and 
transactions are for instance [5,3], or [4,3,1]. As second objective we maximize
the minimal transaction in the chosen subset. 

In general we have to define a mapping 'selection_value' which maps a 
specific selection/subset represented as boolean array to a list of values to be minimized. 

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Subset.adoc for a detailed description.
"""

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.

import numpy as np
from scipy.optimize import Bounds 
from fcmaes import mode, modecpp

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

# replace with your mapping selection -> value
class transaction_value():
    
    def __init__(self, transactions, payments):
        self.transactions = transactions
        self.sum_payments = sum(payments)
        
    def __call__(self, selection):
        trs = self.transactions[selection]
        return abs(sum(trs) - self.sum_payments), -min(trs)
    
class fitness():
    
    def __init__(self, selection_value, dim):
        self.selection_value = selection_value
        self.bounds = Bounds([0]*dim, [1.99999999]*dim)  
        
    def selected(self, x):
        return x.astype(int)
    
    # all decision variables are in the [0,2[ interval and mapped to a boolean array. 
    def __call__(self, x):
        return self.selection_value(x.astype(int).astype(bool))

def optimize(fitness, num_retries = 32):
    nobj = 2
    ncon = 0
    xs, ys = modecpp.retry(mode.wrapper(fitness, nobj), nobj, ncon, 
                           fit.bounds, num_retries=num_retries, popsize = 500, 
                           max_evaluations = 100000, nsga_update = True, 
                           workers=32)    
    # show the best results
    for i in range(len(xs)):
        if ys[i][0] > 10: break
        print(i+1, ") Optimal Objective values: ", ys[i])
        print(fitness.selected(xs[i]))

if __name__ == '__main__':
    seed = 13
    rng = np.random.default_rng(seed)   
    transactions= rng.integers(100, 2500, 1000) / 100  
    payments = rng.integers(10, 50, 100)    
    selection_value = transaction_value(transactions, payments)    
    fit = fitness(selection_value, len(transactions))
    optimize(fit, num_retries=32)
