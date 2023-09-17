# HBV Rainfall-Runoff Model 
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Water.adoc
# See http://www.sciencedirect.com/science/article/pii/S0309170812000073
# See https://github.com/jdherman/awr-hbv-benchmark
# See also https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Diversity.adoc

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import numpy as np
import os, sys, time
import ctypes as ct
from numpy.random import Generator, MT19937
from fcmaes.evaluator import Evaluator
from fcmaes import moretry
import multiprocessing as mp
from fcmaes.optimizer import dtime
from scipy.optimize import Bounds
from fcmaes.optimizer import de_cma, Bite_cpp, Cma_cpp, LDe_cpp, dtime,  De_cpp, random_search, wrapper, logger
from fcmaes import moretry, retry, mode, modecpp, decpp, de, moretry#, modec 
from fcmaes import diversifier, mapelites
from scipy.optimize import Bounds

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

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
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
            
    def __call__(self, x):
        x = np.array(x)
        y = np.empty(nobj)
        x_p = x.ctypes.data_as(ct.POINTER(ct.c_double))
        y_p = y.ctypes.data_as(ct.POINTER(ct.c_double))  
        if self.hbv is None:
            self.hbv = hbv_C()   
        fitness_hbv_C(self.hbv, x_p, y_p)
        return np.array(y)
    
    def qd_fitness(self, x):      
        y = self.__call__(x)
        b = y.copy()
        y = (y - self.qd_bounds.lb) / (self.qd_bounds.ub - self.qd_bounds.lb)
        ws = sum(y)
        if ws < self.best_y.value:
            self.best_y.value = ws
            print(f'{ws:.3f} { list(b) }')            
        return ws, b  
    
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
    
def optimize_mo():
    try:      
        problem = hbv()
        
        store = mode.store(dim, nobj, 2048)
        fun = mode.wrapper(problem, nobj, store, plot=True, interval = 1000000, name='mode256.16')
        
        # parallel function evaluation
        # modecpp.minimize, the C++ variant, works only with workers=1 because of limitations of the
        # used parallelization mechanism in combination with the way the objective function is initialized.
         
        # mode.minimize(fun, nobj, 0, problem.bounds, popsize = 256, 
        #             max_evaluations = 2000000, nsga_update=True, workers=16)

        # parallel optimization retry
        
        modecpp.retry(fun, nobj, 0, 
                      problem.bounds, num_retries=32, popsize = 256, 
                  max_evaluations = 500000, nsga_update = True, workers=32)
        
        
    except Exception as ex:
        print(str(ex))  

def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    x = ys[:, 0]; y = ys[:, 2]; z = ys[:, 1]
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

def plot_archive(archive, problem):
    si = archive.argsort()
    ysp = []
    descriptions = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    xs = archive.get_xs()[si]
    yall = []
    for i in range(len(si)):
        if ys[i] < np.inf: # throw out invalid
            ysp.append(descriptions[i])
    ysp = np.array(ysp)
    plot3d(ysp, "hbv_nd", 'f1', 'f3', 'f2')
                
def optimize_qd():
    problem = hbv()
    problem.qd_dim = 4
    problem.qd_bounds = Bounds(np.array([0.2, 0.7, 0, 0]), 
                               np.array([0.6, 1.3, 0.18, 0.6])) 
    name = 'hbv_nd'
    opt_params0 = {'solver':'elites', 'popsize':64}
    
    #opt_params1 = {'solver':'CMA_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem.qd_fitness, problem.qd_dim, interval=200000, save_interval=5000000), 
         problem.bounds, problem.qd_bounds, opt_params=[opt_params0, opt_params1], max_evals=12000000)
    # archive = mapelites.load_archive(name, problem.bounds, problem.qd_bounds)    
    print('final archive:', archive.info())
    archive.save(name)
    plot_archive(archive, problem)

if __name__ == '__main__':
    # optimize_mo()
    optimize_qd()
     
