# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See  https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Damp.adoc 

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import time, sys, warnings, math

from fcmaes import retry
from fcmaes.optimizer import De_cpp, wrapper
from fcmaes import diversifier, mapelites
import numpy as np
from scipy.integrate import ode
from scipy.optimize import Bounds

import ctypes as ct
import multiprocessing as mp 

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# Numpy based integration
def spring(t, y):
    x1, x2, alpha = y 
    return [x2, -x1 + alpha, 0]

def integrator():
    I = ode(spring)
    I.set_integrator("dopri5", nsteps=1000, rtol=1e-6, atol=1e-6)
    return I

def integrate(I, t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return I.integrate(t)

# C-based integration    
def integrate_C(y, dt, alpha, step):
    try:
        array_type = ct.c_double * y.size     
        ry = integrateDamp_C(array_type(*y), alpha, dt, step)
        y = np.array(np.fromiter(ry, dtype=np.float64, count=y.size))
        freemem(ry)
        return y
    except Exception as e:
        return None # fail

max_alpha = 0.1 
max_time = 40
    
class fitness(object):

    def __init__(self, dim):
        self.dim = dim 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.bounds = Bounds([0]*dim, [1]*dim)
        self.qd_dim = 2
        self.qd_bounds = Bounds([10, 0.5], [40, 1.5])

    def __call__(self, X):
        n = int(self.dim/2)
        dt = 2*max_time/n
        dts = X[:n]*dt 
        alphas = X[n:]*2*max_alpha - max_alpha         
        y = np.array([1,0])
        for i in range(n):
            y = integrate_C(y, dts[i], alphas[i], 0.1)
        return abs(y[0])+abs(y[1])     

    def qd_fit(self, x): 
        y = self(x)
        n = int(self.dim/2)
        dt = 2*max_time/n
        dts = x[:n]*dt 
        alphas = x[n:]*2*max_alpha - max_alpha
        dtsum = np.sum(dts)
        energy = np.sum(np.multiply(dts, abs(alphas)))
        b = np.array([dtsum, energy])
        if y < self.best_y.value:
            self.best_y.value = y
            print(f'{y:.3f} { list(b) }')            
        return y, b  

def parallel_retry(dim, opt = De_cpp(20000)):
    fit = fitness(dim)
    return retry.minimize(wrapper(fit), fit.bounds, optimizer=opt, num_retries=32)

def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    x = ys[:, 0]; y = ys[:, 1]; z = ys[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot()     
    img = ax.scatter(x, y, s=4, c=z, cmap='rainbow')
    cbar = fig.colorbar(img)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)
    cbar.set_label(zlabel)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

def plot_archive(problem, archive = None):
    name = 'damp_nd'
    if archive is None:
        archive = mapelites.load_archive(name, problem.bounds, problem.qd_bounds)
    si = archive.argsort()
    ysp = []
    beh = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    lb = problem.qd_bounds.lb
    ub = problem.qd_bounds.ub
    #xs = archive.get_xs()[si]
    for i in range(len(si)):
        if ys[i] < 1.0: # throw out invalid
            b = beh[i]
            if np.any(np.greater(b, ub)) or np.any(np.greater(lb, b)):
                continue
            ysp.append([b[0], b[1], ys[i]])
    ysp = np.array(ysp)
    plot3d(ysp, name, 'time', 'energy', 'amplitude')
                
def optimize_qd(dim):
    problem = fitness(dim)
    name = 'damp_nd'
    opt_params0 = {'solver':'elites', 'popsize':512}
    #opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    #opt_params1 = {'solver':'DE_CPP', 'max_evals':6000, 'popsize':32, 'stall_criterion':3}
    #opt_params1 = {'solver':'CMA_CPP', 'max_evals':6000, 'pnp.zeros(10)opsize':32, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem.qd_fit, problem.qd_dim, interval=200000, save_interval=12000000), 
         problem.bounds, problem.qd_bounds, opt_params=[opt_params0], max_evals=30000000) 
    print('final archive:', archive.info())
    archive.save(name)
    plot_archive(problem, archive)

from fcmaes.cmaescpp import libcmalib
integrateDamp_C = libcmalib.integrateDamp_C
integrateDamp_C.argtypes = [ct.POINTER(ct.c_double), ct.c_double, ct.c_double, ct.c_double]
integrateDamp_C.restype = ct.POINTER(ct.c_double)   
freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]
    
if __name__ == '__main__':    
    dim = 12
    # apply a QD algorithm
    optimize_qd(dim)
    # plot the result
    plot_archive(fitness(dim))
    
    # lets find the best solution
    #ret = parallel_retry(dim)
 
