
import math
import numpy as np
import os
from scipy.optimize import OptimizeResult, Bounds
from numpy.random import MT19937, Generator
from fcmaes.evaluator import _get_bounds, _fitness, serial, parallel

from typing import Optional, Callable, Union, Dict
from numpy.typing import ArrayLike

""" Numpy based implementation of Fast Moving Natural Evolution Strategy 
    for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
    Derived from https://github.com/nomuramasahir0/crfmnes .
"""

# evaluation value of the infeasible solution
INFEASIBLE = np.inf

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             bounds: Optional[Bounds] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[float] = 0.3,
             popsize: Optional[int] = 32,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = None,
             stop_fitness: Optional[float] = -np.inf,
             is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None,
             rg: Optional[Generator] = Generator(MT19937()),
             runid: Optional[int] = 0,
             normalize: Optional[bool] = False,
             options: Optional[Dict] = {}
             ) -> OptimizeResult:       
    """Minimization of a scalar function of one or more variables using CMA-ES.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (n,)
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.  
    input_sigma : ndarray, shape (n,) or scalar
        Initial step size.
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.     
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to 
        decide when to terminate. 
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the optimization run.
    normalize : boolean, optional
        if true pheno -> geno transformation maps arguments to interval [-1,1]  
    options : dict, optional
   
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object"""

    cr = CRFMNES(None, bounds, x0, input_sigma, popsize, 
                 max_evaluations, stop_fitness, is_terminate, runid, normalize, options, rg, workers, fun)
    
    cr.optimize()

    return OptimizeResult(x=cr.f.decode(cr.x_best), fun=cr.f_best, nfev=cr.no_of_evals, 
                          nit=cr.g, status=cr.stop, 
                          success=True)

class CRFMNES:
    
    def __init__(self, 
                dim = None, 
                bounds: Optional[Bounds] = None, 
                x0: Optional[ArrayLike] = None,
                input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
                popsize: Optional[int] = 32,  
                max_evaluations: Optional[int] = 100000, 
                stop_fitness: Optional[float] = -np.inf, 
                is_terminate: Optional[bool] = None, 
                runid: Optional[int] = 0, 
                normalize: Optional[bool] = False,
                options: Optional[Dict] = {}, 
                rg: Optional[Generator] = Generator(MT19937()), 
                workers: Optional[int] = None, 
                fun: Optional[Callable[[ArrayLike], float]] = lambda x: 0): 
        
        if popsize is None:
            popsize = 32         
        if popsize % 2 == 1: # requires even popsize
            popsize += 1
        if dim is None:
            if not x0 is None: dim = len(x0)
            else: 
                if not bounds is None: dim = len(bounds.lb)
        lower, upper, guess = _get_bounds(dim, bounds, x0, rg) 
        self.fun = serial(fun) if (workers is None or workers <= 1) else parallel(fun, workers)  
        self.f = _fitness(self.fun, lower, upper, normalize)       
        if options is None:
            options = {}
        if not lower is None:
            options['constraint'] = [ [lower[i], upper[i]] for i in range(dim)]   
        self.constraint = options.get('constraint', [[-np.inf, np.inf] for _ in range(dim)])
        if 'seed' in options.keys():
            np.random.seed(options['seed'])
        sigma = input_sigma
        if not np.isscalar(sigma):
            sigma = np.mean(sigma)         
        self.m = np.array([self.f.encode(guess)]).T

        self.dim = dim
        self.sigma = sigma
        self.popsize = popsize
              
        self.max_evaluations = max_evaluations
        self.stop_fitness = stop_fitness
        self.is_terminate = is_terminate
        self.rg = rg
        self.runid = runid

        self.v = options.get('v', self.rg.normal(0,1,(dim, 1)) / np.sqrt(dim))
        
        self.D = np.ones([dim, 1])
        self.penalty_coef = options.get('penalty_coef', 1e5)
        self.use_constraint_violation = options.get('use_constraint_violation', True)

        self.w_rank_hat = (np.log(self.popsize / 2 + 1) - np.log(np.arange(1, self.popsize + 1))).reshape(self.popsize, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.popsize)
        self.mueff = 1 / ((self.w_rank + (1 / self.popsize)).T @ (self.w_rank + (1 / self.popsize)))[0][0]
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
        self.c1_cma = 2. / (math.pow(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.pc = np.zeros([self.dim, 1])
        self.ps = np.zeros([self.dim, 1])
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(self.popsize / self.dim)) * math.sqrt(
            lambF / self.popsize)
        self.w_dist_hat = lambda z, lambF: exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.dim + 20.) / (self.dim + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.dim + 10.) / (self.dim + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6 * (lambF / self.popsize)
        self.eta_B = lambda lambF: np.tanh((min(0.02 * lambF, 3 * np.log(self.dim)) + 5) / (0.23 * self.dim + 25))

        self.g = 0
        self.no_of_evals = 0
        self.iteration = 0
        self.stop = 0

        self.idxp = np.arange(self.popsize / 2, dtype=int)
        self.idxm = np.arange(self.popsize / 2, self.popsize, dtype=int)
        self.z = np.zeros([self.dim, self.popsize])

        self.f_best = float('inf')
        self.x_best = np.empty(self.dim)

    def __del__(self):
        if isinstance(self.fun, parallel):
            self.fun.stop()
        
    def calc_violations(self, x):
        violations = np.zeros(self.popsize)
        for i in range(self.popsize):
            for j in range(self.dim):
                violations[i] += (- min(0, x[j][i] - self.constraint[j][0]) + max(0, x[j][i] - self.constraint[j][1])) * self.penalty_coef
        return violations

    def optimize(self) -> int:
        # -------------------- Generation Loop --------------------------------
        while True:
            if self.no_of_evals > self.max_evaluations:
                break
            if self.stop != 0:
                break
            try:
                x = self.ask()
                y = self.f.values(self.f.decode(self.f.closestFeasible(x)))
                self.tell(y)
                if self.stop != 0:
                    break 
            except Exception as ex:
                self.stop = -1
                break

    def ask(self) -> np.ndarray:
        d = self.dim
        popsize = self.popsize
        zhalf = self.rg.normal(0,1,(d, int(popsize / 2)))  # dim x popsize/2
        self.z[:, self.idxp] = zhalf
        self.z[:, self.idxm] = -zhalf
        self.normv = np.linalg.norm(self.v)
        self.normv2 = self.normv ** 2
        self.vbar = self.v / self.normv
        self.y = self.z + ((np.sqrt(1 + self.normv2) - 1) * (self.vbar @ (self.vbar.T @ self.z)))
        self.x = self.m + (self.sigma * self.y) * self.D
        return self.x.T

    def tell(self, evals_no_sort: np.ndarray) -> int:
        violations = np.zeros(self.popsize)
        if self.use_constraint_violation:
            violations = self.calc_violations(self.x)
            sorted_indices = sort_indices_by(evals_no_sort + violations, self.z)
        else:
            sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]
        f_best = evals_no_sort[best_eval_id]
        x_best = self.x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = self.y[:, sorted_indices]
        x = self.x[:, sorted_indices]

        self.no_of_evals += self.popsize
        self.g += 1
 
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best           
            # print(self.no_of_evals, self.g, self.f_best)

        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(evals_no_sort < np.finfo(float).max)

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        f1 =  self.h_inv * min(1., math.sqrt(self.popsize / self.dim)) * math.sqrt(lambF / self.popsize)        
        w_tmp = self.w_rank_hat * np.exp(np.linalg.norm(self.z, axis = 0) * f1).reshape((self.popsize,1))
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.popsize
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = self.normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x popsize+1
        yy = exY * exY  # dim x popsize+1
        ip_yvbar = self.vbar.T @ exY
        yvbar = exY * self.vbar  # dim x popsize+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + self.normv2
        vbarbar = self.vbar * self.vbar
        alphavd = min(1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + self.normv2))  # scalar
        
        t = exY * ip_yvbar - self.vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x popsize+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.dim, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - self.normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.dim, self.popsize + 1])  # dim x popsize+1
        ip_vbart = self.vbar.T @ t  # 1 x popsize+1
 
        s_step2 = s_step1 - alphavd / gammav * ((2 + self.normv2) * (t * self.vbar) - self.normv2 * vbarbar @ ip_vbart)  # dim x popsize+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x popsize+1
        
        div = 1 + b * vbarbar.T @ invHvbarbar
        if np.amin(abs(div)) == 0:
            return -1
        
        s = (s_step2 * invH) - b / div * invHvbarbar @ ip_s_step2invHvbarbar  # dim x popsize+1
        ip_svbarbar = vbarbar.T @ s  # 1 x popsize+1
        t = t - alphavd * ((2 + self.normv2) * (s * self.vbar) - self.vbar @ ip_svbarbar)  # dim x popsize+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)]).reshape(1, 1),
                        axis=0)  # popsize+1 x 1
        self.v = self.v + (t @ exw) / self.normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        if np.amin(self.D) < 0:
            return -1

        nthrootdetA = exp(np.sum(np.log(self.D)) / self.dim + np.log(1 + (self.v.T @ self.v)[0][0]) / (2 * self.dim))
         
        self.D = self.D / nthrootdetA
        
        # update sigma
        G_s = np.sum((self.z * self.z - np.ones([self.dim, self.popsize])) @ weights) / self.dim
        self.sigma = self.sigma * exp(eta_sigma / 2 * G_s)
        return self.stop

    def population(self) -> np.ndarray:
        return self.x

    def result(self) -> OptimizeResult:
        return OptimizeResult(x=self.x_best, fun=self.f_best, nfev=self.no_of_evals, 
                              nit=self.g, status=self.stop, success=True)
        
def exp(a):
    return math.exp(min(100, a)) # avoid overflow

def get_h_inv(dim):
    f = lambda a, b: ((1. + a * a) * exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * exp(a * a / 2.) * (3. + a * a)
    h_inv = 1.0
    while abs(f(h_inv, dim)) > 1e-10:
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
    return h_inv

def sort_indices_by(evals, z):
    lam = len(evals)
    evals = np.array(evals)
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != INFEASIBLE)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == INFEASIBLE)[0]]
        distances = np.sum(infeasible_z ** 2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices[no_of_feasible_solutions:] = infeasible_indices[indices_sorted_by_distance]
    return sorted_indices
