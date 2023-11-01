# Adapted from https://github.com/yunshengtian/DGEMO/blob/master/problems/re.py
# See Tanabe, Ryoji, and Hisao Ishibuchi. "An easy-to-use real-world multi-objective optimization problem suite." Applied Soft Computing (2020): 106078.
# https://arxiv.org/abs/2009.12867

# Modified to work with https://numba.pydata.org/
# to measure algorithm overhead. 
# Returns constraints unmodified, not as weighted sum
# On an AMD 5950x all problems are solved in < 2 sec 

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import numpy as np    
from fcmaes import mode, modecpp, decpp, de, moretry
from scipy.optimize import Bounds
from numba import njit

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

@njit(fastmath=True)  
def closest_value(arr, val):
    '''
    Get closest value to val in arr
    '''
    return arr[np.argmin(np.abs(arr - val))]

@njit(fastmath=True)  
def div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return x1/x2 if x2 != 0 else 0

@njit(fastmath=True)  
def re_1(x):                
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    F = 10
    E = 2e5
    L = 200
    
    f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
    f2 = (F * L) / E * (div(2.0, x1) + div(2.0 * np.sqrt(2.0), x2) - div(2.0 * np.sqrt(2.0), x3) + div(2.0, x4))

    return np.array([f1,f2])

@njit(fastmath=True)  
def re_2(x, feasible_values): 
    x1, x2, x3 = x[0], x[1], x[2]
    x1 = closest_value(feasible_values, x1)

    f1 = (29.4 * x1) + (0.6 * x2 * x3)

    g = np.array(
        [(x1 * x3) - 7.735 * div((x1 * x1), x2) - 180.0, 
        4.0 - div(x3, x2)
    ])
    g = np.maximum(g*-1, 0)

    return np.array([f1] + list(g))

@njit(fastmath=True)  
def re_3(x): 
    x1, x2 = x[0], x[1]

    f1 = x1 + (120 * x2)

    E = 700000
    sigmaBMax = 700
    tauMax = 450
    deltaMax = 1.5
    sigmaK = (E * x1 * x1) / 100
    sigmaB = div(4500, (x1 * x2))
    tau = div(1800, x2)
    delta = div(56.2 * 10000, E * x1 * x2 * x2)

    g = np.array([
        1 - (sigmaB / sigmaBMax),
        1 - (tau / tauMax),
        1 - (delta / deltaMax),
        1 - div(sigmaB, sigmaK)
    ])
    g = np.maximum(g*-1, 0)
 
    return np.array([f1] + list(g))
        
@njit(fastmath=True)  
def re_4(x): 
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    P = 6000
    L = 14
    E = 30 * 1e6
    G = 12 * 1e6
    tauMax = 13600
    sigmaMax = 30000

    f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
    f2 = div(4 * P * L * L * L, E * x4 * x3 * x3 * x3)

    M = P * (L + (x2 / 2))
    tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
    R = np.sqrt(tmpVar)
    tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
    J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

    tauDashDash = div(M * R, J)
    tauDash = div(P, np.sqrt(2) * x1 * x2)
    tmpVar = tauDash * tauDash + div((2 * tauDash * tauDashDash * x2), (2 * R)) + (tauDashDash * tauDashDash)
    tau = np.sqrt(tmpVar)
    sigma = div(6 * P * L, x4 * x3 * x3)
    tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
    tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
    PC = tmpVar * (1 - tmpVar2)

    g = np.array([
        tauMax - tau,
        sigmaMax - sigma,
        x4 - x1,
        PC - P
    ])
    g = np.maximum(g*-1, 0)

    return np.array([f1,f2] + list(g))

@njit(fastmath=True)  
def re_5(x): 
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    f2 = div((9.82 * 1e6) * (x2 * x2 - x1 * x1), x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

    g = np.array([
        (x2 - x1) - 20.0,
        0.4 - div(x3, (3.14 * (x2 * x2 - x1 * x1))),
        1.0 - div(2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1), np.power((x2 * x2 - x1 * x1), 2)),
        div(2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1), x2 * x2 - x1 * x1) - 900.0
    ])
    g = np.maximum(g*-1, 0)

    return np.array([f1,f2] + list(g))

@njit(fastmath=True)  
def re_6(x): 
    x1, x2, x3, x4 = np.round(x[0]), np.round(x[1]), np.round(x[2]), np.round(x[3])

    f1 = np.abs(6.931 - (div(x3, x1) * div(x4, x2)))
    f2 = np.max(np.array([x1, x2, x3, x4]))
    
    g = np.array([0.5 - (f1 / 6.931)])
    g = np.maximum(g*-1, 0)
    
    return np.array([f1,f2] + list(g))

@njit(fastmath=True)  
def re_7(x): 
    xAlpha, xHA, xOA, xOPTT = x[0], x[1], x[2], x[3]

    f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
    f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
    f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

    return np.array([f1,f2,f3])


class RE1(object):
    '''
    Four bar truss design
    NOTE: the provided true pareto front approximation of this problem might be wrong
    '''
    n_var = 4
    n_obj = 2
    n_con = 0
    xl = [1, np.sqrt(2), np.sqrt(2), 1]
    xu = [3, 3, 3, 3]

    def __call__(self, x):
        return re_1(x)

class RE2(object):
    '''
    Reinforced concrete beam design
    '''
    n_var = 3
    n_obj = 1
    n_con = 2
    xl = [0.2, 0, 0]
    xu = [15, 20, 40]

    feasible_values = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0])

    def __call__(self, x):
        return re_2(x, self.feasible_values)
    
class RE3(object):
    '''
    Hatch cover design
    '''
    n_var = 2
    n_obj = 1
    n_con = 4
    xl = [0.5, 0.5]
    xu = [4, 50]

    def __call__(self, x):
        return re_3(x)
        
class RE4(object):
    '''
    Welded beam design
    '''
    n_var = 4
    n_obj = 2
    n_con = 4
    xl = [0.125, 0.1, 0.1, 0.125]
    xu = [5, 10, 10, 5]

    def __call__(self, x):
        return re_4(x)

class RE5(object):
    '''
    Disc brake design
    '''
    n_var = 4
    n_obj = 2
    n_con = 4
    xl = [55, 75, 1000, 11]
    xu = [80, 110, 3000, 20]

    def __call__(self, x):
        return re_5(x)

class RE6(object):
    '''
    Gear train design
    '''
    n_var = 4
    n_obj = 2
    n_con = 1
    xl = [12] * 4
    xu = [60] * 4

    def __call__(self, x):
        return re_6(x)

class RE7(object):
    '''
    Rocket injector design
    '''
    n_var = 4
    n_obj = 3
    n_con = 0
    xl = [0] * 4
    xu = [1] * 4

    def __call__(self, x):
        return re_7(x)

from pymoo.core.problem import ElementwiseProblem 
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation    
from pymoo.factory import get_termination

class RE1_pymoo(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([1, np.sqrt(2), np.sqrt(2), 1]),
                         xu=np.array([3, 3, 3, 3]), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        y = re_1(x)
        out["F"] = [y[0], y[1]]

class RE4_pymoo(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=4,
                         xl=np.array([0.125, 0.1, 0.1, 0.125]),
                         xu=np.array([5, 10, 10, 5]), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        y = re_4(x)
        out["F"] = [y[0], y[1]]
        out["G"] = [y[2], y[3], y[4], y[5]]

class RE5_pymoo(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=4,
                         xl=np.array([55, 75, 1000, 11]),
                         xu=np.array([80, 110, 3000, 20]), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        y = re_5(x)
        out["F"] = [y[0], y[1]]
        out["G"] = [y[2], y[3], y[4], y[5]]
        
class RE6_pymoo(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([12] * 4),
                         xu=np.array([60] * 4), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        y = re_6(x)
        out["F"] = [y[0], y[1]]
        out["G"] = [y[2]]

def check_pymoo():
    
    problem = RE5_pymoo()

    algorithm = NSGA2(
        pop_size=64,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )    
        
    termination = get_termination("n_gen", 5000)
    
    from pymoo.optimize import minimize
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=False,
                   verbose=False)
    
    X = res.X
    F = res.F
    
    import matplotlib.pyplot as plt
    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()

def main():
    
    # check_pymoo()
    # sys.exit()
    
    problems =  [RE1(),RE2(),RE3(),RE4(),RE5(),RE6(),RE7()] 
    names =  ["RE1","RE2","RE3","RE4","RE5","RE6","RE7"] 

    for p in range(7):
        problem = problems[p]
        pname = "nsga_on100k_" + names[p]
        
        # x, y = modecpp.minimize(mode.wrapper(problem, problem.n_obj), problem.n_obj, 0,
        #             Bounds(problem.xl,problem.xu), popsize = 64, 
        #             max_evaluations = 100000, 
        #             nsga_update=True, 
        #             plot_name = pname, workers=16)
        
        x, y = modecpp.retry(mode.wrapper(problem, problem.n_obj), problem.n_obj, problem.n_con,
                     Bounds(problem.xl,problem.xu), popsize = 64, 
                     max_evaluations = 100000, 
                     nsga_update=True, num_retries = 16,
                     workers=16)

        np.savez_compressed(pname, xs=x, ys=y)
        moretry.plot(pname, problem.n_con, x, y, all=False, interp=True, plot3d=True)
        
if __name__ == '__main__':
    main()