# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
from fcmaes.testfun import Wrapper, Rosen, Rastrigin, Eggholder
from fcmaes import cmaes, cmaescpp, retry, advretry
from scipy.optimize import OptimizeResult

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
    assert(max_eval + popsize > ret.nfev) # too much function calls
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
    assert(max_eval + popsize > ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_ask_tell():    
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000   
    limit = 0.00001 
    popsize = 31  
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        es = cmaes.Cmaes(testfun.bounds,
                popsize = popsize, input_sigma = sdevs)       
        iters = 50000 // popsize
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
    assert(max_eval + popsize > ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_cpp():
    if not sys.platform.startswith('linux'):
        return
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
    assert(max_eval + popsize > ret.nfev) # too much function calls 
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_parallel():
    # parallel execution slows down the test since we are using a test function
    # which is very fast to evaluate
    # popsize defines the maximal number of used threads
    
    #windows cannot pickle function objects
    if sys.platform.startswith('windows'):
        return

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
                       popsize=popsize, is_parallel=True)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize > ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

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
    assert(max_eval + popsize > ret.nfev) # too much function calls
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_retry():
    #windows cannot pickle function objects
    if sys.platform.startswith('windows'):
        return

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
    #windows cannot pickle function objects
    if sys.platform.startswith('windows'):
        return

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
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned
 
def almost_equal(X1, X2):
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
