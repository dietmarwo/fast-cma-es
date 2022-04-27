import numpy as np
import os, sys, time
import ctypes as ct
from numpy.random import Generator, MT19937
from fcmaes.evaluator import Evaluator
from fcmaes import moretry
import multiprocessing as mp
from fcmaes.optimizer import logger, dtime
from scipy.optimize import Bounds
from fcmaes.optimizer import de_cma, Bite_cpp, Cma_cpp, LDe_cpp, dtime,  De_cpp, random_search, wrapper, logger
from fcmaes import moretry, retry, mode, modecpp, decpp, de, moretry#, modec 

basepath = os.path.dirname(os.path.abspath(__file__))
libhbv = ct.cdll.LoadLibrary(basepath + '/../../fcmaes/lib/libhbv.so')  

    
hbv_C = libhbv.hbv_C
hbv_C.argtypes = []
hbv_C.restype = ct.c_long

fitness_hbv_C = libhbv.fitness_hbv_C
fitness_hbv_C.argtypes = [ct.c_long, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
 
nobj = 4

bounds = [
       (0.0, 100.0), #L (mm) 
       (0.5, 20.0), #K0 (d)
       (1.0, 100.0), #K1 (d)
       (10.0, 20000.0), #K2 (d)
       (0.0, 100.0), #Perc (mm/d)
       (0.3, 1.0), #LP (-)
       (0.0, 2000.0), #Fcap (mm)
       (0.0, 7.0), #B (-)
       (24.0, 120.0), #MaxBas (d)
       (-3.0, 3.0), #TT (C)
       (0.0, 20.0), #DDF (mm/C*d)
       (0.0, 1.0), #CFR (-)
       (0.0, 0.8), #CWH (-)
       (0.0, 7.0), #TTI (C)   
    ]

dim = len(bounds)
lb = np.array([b[0] for b in bounds])
ub = np.array([b[1] for b in bounds])

class hbv(object):
    
    def __init__(self):
        self.bounds = Bounds(lb, ub)
        self.hbv = None
    
    def __call__(self, x):
        x = np.array(x)
        y = np.empty(nobj)
        x_p = x.ctypes.data_as(ct.POINTER(ct.c_double))
        y_p = y.ctypes.data_as(ct.POINTER(ct.c_double))  
        if self.hbv is None:
            self.hbv = hbv_C()   
        fitness_hbv_C(self.hbv, x_p, y_p)
        return np.array(y)
    
def check_pymoo(index):

    from pymoo.core.problem import ElementwiseProblem 
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.factory import get_sampling, get_crossover, get_mutation    
    from pymoo.factory import get_termination
    from multiprocessing.pool import ThreadPool
    from pymoo.operators.sampling.lhs import LHS
    
    store = mode.store(dim, nobj, 2048)
        
    wrapped = mode.wrapper(hbv(), nobj, store, plot=True, interval = 10000, 
                           name='nsga256_' + str(index))     
            
    class MyProblem(ElementwiseProblem):
    
        def __init__(self, **kwargs):
            super().__init__(n_var=dim,
                             n_obj=nobj,
                             n_constr=0,
                             xl=np.array(lb),
                             xu=np.array(ub), **kwargs)
    
        def _evaluate(self, x, out, *args, **kwargs):   
            y = wrapped(x)
            out["F"] = y[:nobj]
            out["G"] = y[nobj:]

    pool = ThreadPool(8)
    #pool = multiprocessing.Pool(32)
    problem = MyProblem()

    algorithm = NSGA2(
        pop_size=256,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),        
        eliminate_duplicates=True
    )    
    
    from pymoo.optimize import minimize
    import matplotlib.pyplot as plt
        
    res = minimize(problem,
                   algorithm,
                   get_termination("n_gen", 10000),
                   save_history=True,
                   verbose=False)

    X = res.X
    F = res.F
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    #plt.show()
    plt.savefig('NSGSII256-objective-space'+ str(index) + '.png')
    plt.clf() 
    sys.exit()

    
def main():
    try:      
        # check_pymoo(1)

        problem = hbv()
        
        store = mode.store(dim, nobj, 2048)
        fun = mode.wrapper(problem, nobj, store, plot=True, interval = 1000000, name='mode256.16')
        
        # mode.minimize(fun, nobj, 0, problem.bounds, popsize = 256, 
        #             max_evaluations = 2000000, nsga_update=False, workers=16)

        modecpp.retry(fun, nobj, 0, 
                      problem.bounds, num_retries=32, popsize = 256, 
                  max_evaluations = 500000, nsga_update = False, workers=32)
        
        
    except Exception as ex:
        print(str(ex))  

if __name__ == '__main__':
    main()
     
