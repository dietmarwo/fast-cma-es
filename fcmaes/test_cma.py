# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import multiprocessing as mp
import numpy as np
from scipy.optimize import OptimizeResult
from fcmaes.testfun import Wrapper, Rosen, Rastrigin, Eggholder
from fcmaes import cmaes, de, cmaescpp, gcldecpp, retry, advretry

def test_rastrigin_python():
    popsize = 100
    dim = 3
    testfun = Rastrigin(dim)
    sdevs = [1.0]*dim
    max_eval = 100000

    limit = 0.00001   
    # stochastic optimization may fail the first time
    for _ in range(5):
        # use a wrapper to monitor function evaluations
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
    
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(ret.status == 4) # wrong cma termination code
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_python():
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
    
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_ask_tell():    
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000   
    limit = 0.00001 
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        es = cmaes.Cmaes(testfun.bounds,
                popsize = popsize, input_sigma = sdevs)       
        iters = max_eval // popsize
        for j in range(iters):
            xs = es.ask()
            ys = [wrapper.eval(x) for x in xs]
            stop = es.tell(ys)
            if stop != 0:
                break 
        ret = OptimizeResult(x=es.best_x, fun=es.best_value, 
                             nfev=wrapper.get_count(), 
                             nit=es.iterations, status=es.stop)
        if limit > ret.fun:
            break
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
#     assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_cpp():
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaescpp.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                   max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls 
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_parallel():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 10000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = mp.cpu_count())
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_delayed():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 10000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize, delayed_update=True)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
#     assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
#     assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned

def test_rosen_cpp_parallel():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 10000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaescpp.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = mp.cpu_count())
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_gclde_parallel():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = gcldecpp.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = mp.cpu_count())
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_de():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = de.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = None)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + 2*popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_de_delayed():
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.01   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = de.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
#     assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned

def test_rosen_ask_tell_de():    
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000  
    limit = 0.00001 
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        es = de.DE(dim, testfun.bounds, popsize = popsize)       
        iters = max_eval // popsize
        for j in range(iters):
            xs = es.ask()
            ys = [wrapper.eval(x) for x in xs]
            stop = es.tell(ys, xs)
            if stop != 0:
                break 
        ret = OptimizeResult(x=es.best_x, fun=es.best_value, 
                             nfev=wrapper.get_count(), 
                             nit=es.iterations, status=es.stop)
        if limit > ret.fun:
            break
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + 2*popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
#     assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_python():
    popsize = 1000
    dim = 2
    testfun = Eggholder()
    # use a wrapper to monitor function evaluations
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = -800   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)        
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
   
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_retry():
    dim = 2
    testfun = Eggholder()

    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = retry.minimize(wrapper.eval, testfun.bounds, 
                             num_retries=100)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_advanced_retry():
    dim = 2
    testfun = Eggholder()

    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = advretry.minimize(wrapper.eval, testfun.bounds, 
                                num_retries=300)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned
 
def almost_equal(X1, X2):
    if np.isscalar(X1):
        X1 = [X1]
        X2 = [X2]
    if len(X1) != len(X2):
        return False
    eps = 1E-5
    for i in range(len(X1)):
        a = X1[i]
        b = X2[i]
        if abs(a) < eps or abs(b) < eps:
            if abs(a - b) > eps:
                return False
        else:
            if abs(a / b - 1 > eps):
                return False
    return True
