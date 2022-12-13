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
from fcmaes import diversifier, mapelites
from scipy.optimize import Bounds

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
        self.best_y = mp.RawValue(ct.c_double, np.inf)
                
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
    
    def qd_fitness(self, x):      
        y = self.__call__(x)
        b = y[:nobj].copy()
        constr = np.maximum(y[nobj:], 0) # we are only interested in constraint violations       
        c =  np.amax(constr)
        if c > 0.001: c += 10 
        y = (y[:nobj] - self.qd_bounds.lb) / (self.qd_bounds.ub - self.qd_bounds.lb)
        ws = sum(y) + c
        if ws < self.best_y.value:
            self.best_y.value = ws
            print(f'{ws:.3f} {sum(constr):.3f} { list(b) }')            
        return ws, b        
 
 
def check_pymoo(index):

    from pymoo.core.problem import ElementwiseProblem 
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.factory import get_sampling, get_crossover, get_mutation    
    from pymoo.factory import get_termination, get_reference_directions
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
    plt.titl10000e("Objective Space")
    #plt.show()
    plt.savefig('NSGSII256-objective-space'+ str(index) + '.png')
    plt.clf() 
    sys.exit()
    
def optimize_mo():
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

def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    x = ys[:, 0]; y = ys[:, 4]; z = ys[:, 2]
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
    for i in range(len(si)):
        if ys[i] < np.inf: # throw out invalid
            ysp.append(descriptions[i])
    ysp = np.array(ysp)
    plot3d(ysp, "lrgv_nd", 'x', 'y', 'z')        

def optimize_qd():

    problem = lrgv()
    problem.qd_dim = 5
    problem.qd_bounds = Bounds([0.85E7, -1, 10000, 0, 0], [1.4E7, -0.985, 65000, 65000, 10]) 
    name = 'lrgv_qd'
    opt_params0 = {'solver':'elites', 'popsize':32}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':400, 'popsize':16, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem.qd_fitness, problem.qd_dim, interval=1000, save_interval=200000), 
         problem.bounds, problem.qd_bounds, opt_params=[opt_params0, opt_params1], max_evals=400000)
    
    print('final archive:', archive.info())
    archive.save(name)
    plot_archive(archive, problem)

if __name__ == '__main__':
    # optimize_mo()
    optimize_qd()
     
