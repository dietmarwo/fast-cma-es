"""Find the optimal subset fulfilling any property.

Single objective variant of https://github.com/dietmarwo/fast-cma-es/blob/master/examples/subset_mo.py 

As example we use transactions where a subset is to be matched to a list of payments. 
For transactions: [2,4,5,1,3] and payments: [4, 4] the sum of payments is 8. 
Subsets of transactions that minimize the difference between payments and 
transactions are for instance [5,3], or [4,3,1]. 

In general we have to define a mapping 'selection_value' which maps a 
specific selection/subset represented as boolean array to a value to be minimized. 

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Subset.adoc for a detailed description.
"""

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.

import numpy as np
from scipy.optimize import Bounds 
from fcmaes import retry
from fcmaes.optimizer import wrapper, Bite_cpp

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# replace with your mapping selection -> value
class transaction_value():
    
    def __init__(self, transactions, payments):
        self.transactions = transactions
        self.sum_payments = sum(payments)
        
    def __call__(self, selection):
        return abs(sum(self.transactions[selection]) - self.sum_payments)
    
class fitness():
    
    def __init__(self, selection_value, dim):
        self.selection_value = selection_value
        self.bounds = Bounds([0]*dim, [1.99999999]*dim)  
        
    def selected(self, x):
        return x.astype(int)
    
    # all decision variables are in the [0,2[ interval and mapped to a boolean array. 
    def __call__(self, x):
        return self.selection_value(x.astype(int).astype(bool))

# Multiple optimizations are executed in parallel and all results are collected 
def optimize(fitness, opt, num_retries = 32):
    store = retry.Store(wrapper(fitness), fitness.bounds)
    retry.retry(store, opt.minimize, num_retries)
    xs = store.get_xs()
    ys = store.get_ys()
    # show the best results
    for i in range(len(xs)):
        if ys[i] > 0.001: break
        print(i+1, ") Optimal Objective value: ", ys[i])
        print(fitness.selected(xs[i]))
 
if __name__ == '__main__':
    seed = 13
    rng = np.random.default_rng(seed)   
    transactions= rng.integers(100, 2500, 1000) / 100  
    payments = rng.integers(10, 50, 100)    
    selection_value = transaction_value(transactions, payments)    
    fit = fitness(selection_value, len(transactions))
    # use Bite_cpp(10000) for smaller dimension
    opt = Bite_cpp(50000, popsize=500)
    optimize(fit, opt, num_retries=32)
