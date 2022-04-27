# Lower Rio Grande Valley (LRGV) problem, 5 objective + 3 constraint variant
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Water.adoc
# See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR015976
# See https://github.com/jrkasprzyk/LRGV 

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
from fcmaes import moretry, retry, mode, modecpp, decpp, de, moretry

basepath = os.path.dirname(os.path.abspath(__file__))
liblrgv = ct.cdll.LoadLibrary(basepath + '/../../fcmaes/lib/liblrgv.so')  
    
lrgv_C = liblrgv.lrgv_C
lrgv_C.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
lrgv_C.restype = ct.c_long

fitness_lrgv_C = liblrgv.fitness_lrgv_C
fitness_lrgv_C.argtypes = [ct.c_long, ct.POINTER(ct.c_double),
                           ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
 
dim = 8
nobj = 5
ncon = 3

lb = np.array([0]*3 + [0.1] + [0]*4)
ub = np.array([1]*3 + [0.4] + [3]*4)

class lrgv(object):
    
    def __init__(self):
        self.bounds = Bounds(lb, ub)      
        self.lrgv_p = 0 
        self.name = "lrgvDE"
        
    def __call__(self, x):
        y = np.empty(nobj) # C fitness call returns 6 objectives
        c = np.empty(ncon)
        x_p = x.ctypes.data_as(ct.POINTER(ct.c_double))
        y_p = y.ctypes.data_as(ct.POINTER(ct.c_double))  
        c_p = c.ctypes.data_as(ct.POINTER(ct.c_double))  
        try:     
            if self.lrgv_p == 0:
                args = ["-m","std-io","-b","AllDecAll","-c","ten-year"]
                arguments = [bytes(argsi, 'utf-8') for argsi in args]
                argv = (ct.c_char_p * len(arguments))()
                argv[:] = arguments
                argc = len(arguments)
                self.lrgv_p = lrgv_C(argc, argv)
            fitness_lrgv_C(self.lrgv_p, x_p, y_p, c_p)
        except Exception as ex:
            print(str(ex))     
        return np.array(list(y) + list(np.array(c)*-1.0)) # negate constraints
 
def check_pymoo(index):

    from pymoo.core.problem import ElementwiseProblem 
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.factory import get_sampling, get_crossover, get_mutation    
    from pymoo.factory import get_termination, get_reference_directions
    from pymoo.core.problem import starmap_parallelized_eval
    from multiprocessing.pool import ThreadPool
    from pymoo.operators.sampling.lhs import LHS
    
    lrgv_problem = lrgv()
    
    store = mode.store(dim, nobj, 10240)
        
    wrapped = mode.wrapper(lrgv_problem, nobj, store, plot=True, interval = 500, 
                           name='nsga256_' + str(index))     
            
    class MyProblem(ElementwiseProblem):
    
        def __init__(self, **kwargs):
            super().__init__(n_var=dim,
                             n_obj=nobj,
                             n_constr=ncon,
                             xl=np.array(lb),
                             xu=np.array(ub), **kwargs)
    
        def _evaluate(self, x, out, *args, **kwargs):   
            y = wrapped(x)
            out["F"] = y[:nobj]
            out["G"] = y[nobj:]

    pool = ThreadPool(8)
    #pool = multiprocessing.Pool(32)
    problem = MyProblem()
    #problem = MyProblem(runner=pool.starmap, func_eval=starmap_parallelized_eval)

    #ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    
    # algorithm = CTAEA(ref_dirs=ref_dirs,
    #     sampling=get_sampling("real_random"),
    #     crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    #     mutation=get_mutation("real_pm", eta=20),
    #     eliminate_duplicates=True
    #                   )
    #
    # algorithm = AGEMOEA(
    #     pop_size=768,
    #     n_offsprings=10,
    #     sampling=get_sampling("real_random"),
    #     crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    #     mutation=get_mutation("real_pm", eta=20),
    #     eliminate_duplicates=True        
    #     )

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
  
        problem = lrgv()
        
        store = mode.store(dim, nobj, 10240)
        
        fun = mode.wrapper(problem, nobj, store, plot=True, interval = 5000, name='mode512.16')
        
        # parallel function evaluation
        # modecpp.minimize, the C++ variant, works only with workers=1 because of limitations of the
        # used parallelization mechanism in combination with the way the objective function is initialized.
        
        mode.minimize(fun, 
                         nobj, ncon, problem.bounds, popsize = 512, 
                    max_evaluations = 200000, 
                    nsga_update=False, workers=16)

        # parallel optimization retry
        
        # modecpp.retry(fun, nobj, ncon, 
        #               problem.bounds, num_retries=640, popsize = 256, 
        #           max_evaluations = 3000000, nsga_update = False, workers=32)

        
    except Exception as ex:
        print(str(ex))  

if __name__ == '__main__':
    main()
     
