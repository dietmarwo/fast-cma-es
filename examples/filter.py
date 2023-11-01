# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This example uses the "noisy Travelers Salesman Problem" and applies a machine learning
# approach to avoid unnecessary function calls. Works only with the Python variant of
# differential evolution, both single threaded or with parallel function evaluation. 
# A machine learning based filter should only be used with expensive objective functions. 

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Filter.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import numpy as np
from fcmaes import de
import xgboost
from collections import deque    
from noisy_tsp import TSP, load_tsplib

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# do 'pip install tsplib95'

class filter():
    
    def __init__(self, size, interval, filter_prob = 0.9):
        self.xq = deque(maxlen=size)
        self.yq = deque(maxlen=size)
        self.interval = interval
        self.filter_prob = filter_prob # probability filter is applied
        self.num = 0
        self.model = None
    
    def add(self, x, y):
        self.xq.append(x)
        self.yq.append(y)
        self.num += 1
        if self.num % self.interval == 0:
            try:
                self.learn()
            except Exception as ex:
                print(ex)
    
    def x(self):
        return np.array(self.xq)
    
    def y(self):
        return np.array(self.yq)
        
    def learn(self):
        if self.model is None:
            self.model = xgboost.XGBRegressor(objective='rank:pairwise')
        self.model.fit(self.x(), self.y())
        pass
        
    def is_improve(self, x, x_old, y_old):
        if self.model is None or np.random.random() > self.filter_prob :
            return True
        else:
            try:
                y = self.model.predict([x, x_old])
                return y[0] < y[1]
            except Exception as ex:
                print(ex)
                return True

    def optimize(self, problem):
        
        return de.minimize(problem, 
                dim = problem.d,
            bounds = problem.bounds(), 
            popsize = 16, 
            max_evaluations = 60000, 
            workers = 32,
            filter = self
            # logger = logger()
            )
 
if __name__ == '__main__':
    
    filter = filter(96,32)
    tsp = load_tsplib('data/tsp/br17.tsp')
    filter.optimize(tsp)



