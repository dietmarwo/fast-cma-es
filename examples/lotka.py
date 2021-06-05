
import time, sys, warnings, math

from fcmaes import retry, advretry
from fcmaes.optimizer import logger, de_cma, Bite_cpp, De_cpp, Cma_cpp, LDe_cpp, Minimize, dtime
from fcmaes.de import DE
from fcmaes.cmaes import Cmaes
import numpy as np
from scipy.integrate import ode
from scipy.optimize import Bounds

import ctypes as ct
import multiprocessing as mp 
import pylab as p
from numba.tests.test_array_constants import dt

# Definition of parameters from https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
a = 1.
b = 0.1
c = 1.5
d = b*0.75
pop0 = [10, 5] # initial population 10 rabbits, 5 foxes at t0 = 0
dim = 20 # years
bounds = Bounds([-1]*dim, [1]*dim) # X[i] < 0 means: no fox killing this year

# Lodka Volterra differential equations 
# Propagates a population of x rabbits and y foxes
def lotkavolterra(t, pop, a, b, c, d):
    x, y = pop
    return [a*x - b*x*y, -c*y + d*x*y]

def integrator():
    I = ode(lotkavolterra)
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    # the “dopri5” integrator is reentrant
    I.set_integrator("dopri5", nsteps=1000, rtol=1e-6, atol=1e-6)
    I.set_f_params(a,b,c,d)
    return I

def integrate(I, t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return I.integrate(t)

# maximal rabbit population after dim years of fox killings 
class fitness(object):

    def __init__(self):
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.t0 = time.perf_counter()

    def __call__(self, X):
        ts = []
        for year, x in enumerate(X):
            if x > 0: # should we kill a fox this year? 
                ts.append(year + x) # when exactly?
        I = integrator()
        I.set_initial_value(pop0, 0)
        for i in range(len(ts)):
            pop = integrate(I, ts[i]) # propagate rabbit and fox population to ts[i]      
            pop[1] = max(1, pop[1]-1) # kill one fox, but keep at least one
            I.set_initial_value(pop, ts[i])
        # value is maximal rabbit population during the following 5 years without fox killings
        y = -max([integrate(I, t)[0] for t in np.linspace(dim, dim + 5, 50)])
        # book keeping and logging
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y
            logger().info("nfev = {0}: t = {1:.1f} fval = {2:.3f} fox kill at {3:s} x = {4:s}"
                .format(self.evals.value, dtime(self.t0), y, str([round(t,2) for t in ts[:-1]]), str(list(X))))
        return y     

# parallel optimization with smart boundary management, DE works best
def smart_retry(opt = De_cpp(1500)):
    return advretry.minimize(fitness(), bounds, optimizer=opt, num_retries=50000, max_eval_fac=20)

# parallel independent optimization, BiteOpt works best
def parallel_retry(opt = Bite_cpp(100000, M=8)):
    return retry.minimize(fitness(), bounds, optimizer=opt)

# parallel independent optimization for improvement of an existing solution. Bite_cpp, LDe_cpp and Cma_cpp can be used.
def parallel_improve(opt):
    return retry.minimize(fitness(), bounds, optimizer=opt)

# parallel function evaluation, single optimization, DE works best
def parallel_eval(opt = DE(dim, bounds)):
    return opt.do_optimize_delayed_update(fun=fitness(), max_evals=5000000)

solution = [0.7764942271302568, 9.831131324541304e-13, -0.4392523575954558, 0.9999999991093724, 0.9999999993419174, 0.877806604524956, -0.21969547982373291, 0.9877830923045987, 0.21691094924304902, -0.016089523522436144, 1.0, 0.7622848572479829, -0.0004231871176822595, -0.015617623735551967, -0.9227281069513724, 0.8517521143397784, 8.397851857275901e-19, 1.0, 1.0, 0.1509108812092751]

if __name__ == '__main__':
    print("shoot no fox at all, fitness =", fitness()([-0.5]*dim)) 
    print("shoot a fox every year, fitness =", fitness()([0.5]*dim)) 
    print("best solution, fitness =", fitness()(solution))
    
    # lets find the best solution
    ret = smart_retry()    
    #ret = parallel_retry()
    #ret = parallel_eval()
    #parallel_improve(Bite_cpp(1000000, M=16, guess=sol))
    #parallel_improve(LDe_cpp(1000000, guess=sol))
    #parallel_improve(Cma_cpp(1000000, guess=sol))

    #parallel_retry(opt = Minimize(500000))
