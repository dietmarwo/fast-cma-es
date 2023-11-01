# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Scheduling.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import pandas as pd
import numpy as np
import math, time
from fcmaes import retry, advretry, mode, modecpp
from fcmaes.optimizer import Bite_cpp, Cma_cpp, De_cpp, de_cma, dtime, Dual_annealing, Differential_evolution, Minimize
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
from numba import njit, numba

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

STATION_NUM = 12 # number of dyson ring stations
TRAJECTORY_NUM = 50 # we select 10 mothership trajectories from these trajectories
ASTEROID_NUM = 83454 # number of asteroids

MAX_TIME = 20.0 # mission time in years
WAIT_TIME = 90/365.25 # years, after arrival wait until construction may start
ALPHA = 6.0e-9 # conversion factor time of flight -> arrival mass
#A_DYSON = 1.3197946923098154 # ESAs dyson sphere
A_DYSON = 1.1 # Tsinhuas dyson sphere

DAY = 24 * 3600
YEAR = DAY*365.25

@njit(fastmath=True)        
def select(asteroid, station, trajectory, mass, transfer_start, transfer_time, x):
    trajectories = trajectory_selection(x, TRAJECTORY_NUM)[1] # select 10 trajectories representing the 10 mothership trajectories
    stations = dyson_stations(x, STATION_NUM) # derive dyson_stations targeted at time slots from argument vector
    times = timings(x, STATION_NUM) # derive station build time slot boundaries from argument vector (in years)
    slot_mass = np.zeros(STATION_NUM) # mass sum per time slot
    ast_val = np.zeros(ASTEROID_NUM) # deployed mass for asteroid
    ast_slot = np.zeros(ASTEROID_NUM, dtype=numba.int32) # build time slot used for asteroid
    for i in range(asteroid.size):
        tr = int(trajectory[i]) # current trajectory
        if trajectories[tr] == 0: # trajectory not selected
            continue
        ast_id = int(asteroid[i]) # asteroid transferred
        stat = int(station[i]) # dyson sphere station targeted 
        m = mass[i] # estimated asteroid mass at arrival time 
        time_of_flight = transfer_time[i] # TOF of asteroid transfer
        arrival_time = transfer_start[i] + transfer_time[i] # arrival time of asteroid transfer
        # which station time slot ? 
        for slot in range(STATION_NUM):
            max_time = times[slot+1] # time interval of time slot
            slot_time = times[slot]
            min_time = slot_time + WAIT_TIME # we have to wait 90 days
            if min_time >= MAX_TIME:
                continue
            if arrival_time >= slot_time and arrival_time <= max_time: # inside time slot
                if stat == stations[slot]: # does the station fit?
                    tof = time_of_flight                     
                    #if we have to fly a non optimal transfer, arrival mass is reduced
                    if arrival_time < min_time: # 90 DAYS are not yet over
                        to_add = min_time - arrival_time # add time difference
                        to_add *= math.sqrt(1 + to_add/WAIT_TIME) # add some more time to enable transfer
                        tof += to_add
                    mval = (1.0 - YEAR*tof*ALPHA) * m # estimated asteroid mass at arrival time 
                    if ast_val[ast_id] > 0: # asteroid already transferred                                                
                        old_slot = ast_slot[ast_id]
                        min_mass = np.amin(slot_mass) # greedily replace if current mass is higher
                        old_mass = slot_mass[old_slot] # but never replace at a nearly minimal slot
                        if (old_slot == slot or min_mass < 0.99*old_mass) and ast_val[ast_id] < mval: 
                            # replace with actual transfer, remove old asteroid mass
                            slot_mass[old_slot] -= ast_val[ast_id]                       
                        else: # keep old transfer, don't use the new one
                            mval = 0
                    if mval > 0:  # register actual transfer
                        slot_mass[slot] += mval
                        ast_val[ast_id] = mval
                        ast_slot[ast_id] = slot                  
    slot_mass.sort()
    min_mass = slot_mass[0]
    f = 1.0
    for m in slot_mass:
        # help the optimizer in case the minimum is 0
        min_mass += f*m
        f *= 0.5
    return min_mass, slot_mass

class fitness(object): # the objective function

    def __init__(self, transfers):
        self.evals = mp.RawValue(ct.c_long, 0)  # writable across python processes
        self.best_y = mp.RawValue(ct.c_double, np.inf) # writable across python processes
        self.t0 = time.perf_counter()
        self.transfers = transfers
        self.asteroid = transfers["asteroid"].to_numpy()
        self.station = transfers["station"].to_numpy()  
        self.trajectory = transfers["trajectory"].to_numpy()
        self.transfer_start = transfers["transfer_start"].to_numpy()  
        self.transfer_time = transfers["transfer_time"].to_numpy()
        self.mass = transfers["mass"].to_numpy()          
        self.dv = transfers["dv"].to_numpy()     
        self.trajectory_dv = trajectory_dv(self.asteroid, self.trajectory, self.dv)
        self.nobj = 2
        self.ncon = 0
        
    def __call__(self, x): # single objective      
        # determine the minimal station mass
        min_mass, slot_mass = select(self.asteroid, self.station, self.trajectory, self.mass, 
                        self.transfer_start, self.transfer_time, x) 
        sdv = select_dvs(self.trajectory_dv, x)
        y = -score(min_mass, sdv)
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y   
            trajectories = trajectory_selection(x, TRAJECTORY_NUM)[0] 
            stations = dyson_stations(x, STATION_NUM) 
            times = timings(x, STATION_NUM) 

            sc = score(np.amin(slot_mass), sdv)

            logger().info("evals = {0}: time = {1:.1f} s = {2:.0f} a = {3:.0f} t = {4:s} s = {5:s} b = {6:s} m = {7:s} dv = {8:s}"
                .format(self.evals.value, dtime(self.t0), sc, ast_num(x, self.asteroid, self.trajectory), 
                        str([round(ti,2) for ti in times[1:-1]]), 
                        str([int(si) for si in stations]),
                        str([int(ti) for ti in trajectories]),
                        str([round(mi,2) for mi in slot_mass*1E-15]),
                        str([round(di,2) for di in sdv])
                        ))
        return y    
    
    def fun(self, x): # multiple objectives      
        min_mass, slot_mass = select(self.asteroid, self.station, self.trajectory, self.mass, 
                        self.transfer_start, self.transfer_time, x) 
        sdv = select_dvs(self.trajectory_dv, x)
        scr, dv_val = score_vals(np.amin(slot_mass), sdv)
        y = -scr
        ys = [-min_mass*1E-10, dv_val]
        self.evals.value += 1
        
        if y < self.best_y.value:
            self.best_y.value = y     
            trajectories = trajectory_selection(x, TRAJECTORY_NUM)[0] 
            stations = dyson_stations(x, STATION_NUM) 
            times = timings(x, STATION_NUM) 
            sc = score(np.amin(slot_mass), sdv)
            logger().info("evals = {0}: time = {1:.1f} s = {2:.0f} a = {3:.0f} t = {4:s} s = {5:s} b = {6:s} m = {7:s} dv = {8:s}"
                .format(self.evals.value, dtime(self.t0), -self.best_y.value, ast_num(x, self.asteroid, self.trajectory), 
                        str([round(ti,2) for ti in times[1:-1]]), 
                        str([int(si) for si in stations]),
                        str([int(ti) for ti in trajectories]),
                        str([round(mi,2) for mi in slot_mass*1E-15]),
                        str([round(di,2) for di in sdv])
                        ))
        return ys  
    
    def qd_fun(self, x): # quality diversity    
        _, slot_mass = select(self.asteroid, self.station, self.trajectory, self.mass, 
                        self.transfer_start, self.transfer_time, x) 
        sdv = select_dvs(self.trajectory_dv, x)
        _, dv_val = score_vals(np.amin(slot_mass), sdv)
        sc = score(np.amin(slot_mass), sdv)
        y = -sc
        self.evals.value += 1
        d = np.array([np.amin(slot_mass)*1E-15, dv_val])       
        if y < self.best_y.value:
            self.best_y.value = y     
            trajectories = trajectory_selection(x, TRAJECTORY_NUM)[0] 
            stations = dyson_stations(x, STATION_NUM) 
            times = timings(x, STATION_NUM) 
            logger().info("evals = {0}: time = {1:.1f} s = {2:.0f} a = {3:.0f} t = {4:s} s = {5:s} b = {6:s} m = {7:s} dv = {8:s}"
                .format(self.evals.value, dtime(self.t0), -self.best_y.value, ast_num(x, self.asteroid, self.trajectory), 
                        str([round(ti,2) for ti in times[1:-1]]), 
                        str([int(si) for si in stations]),
                        str([int(ti) for ti in trajectories]),
                        str([round(mi,2) for mi in slot_mass*1E-15]),
                        str([round(di,2) for di in sdv])
                        ))
        return y, d  

    def score(self, x):     
        _, slot_mass = select(self.asteroid, self.station, self.trajectory, self.mass, 
                        self.transfer_start, self.transfer_time, x)
        sdv = select_dvs(self.trajectory_dv, x) 
        return score(np.amin(slot_mass), sdv)
    
def check_pymoo(dim, fit, lb, ub, is_MO):

    from pymoo.core.problem import ElementwiseProblem 
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.factory import get_sampling, get_crossover, get_mutation    
    from pymoo.factory import get_termination
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
    from pymoo.core.problem import starmap_parallelized_eval
    from multiprocessing.pool import ThreadPool
     
    if is_MO: 
        lb[:10] = 0 
        ub[:10] = TRAJECTORY_NUM-1 # integer variables include upper bound 
            
    class MyProblem(ElementwiseProblem):
    
        def __init__(self, **kwargs):
            super().__init__(n_var=dim,
                             n_obj=2,
                             n_constr=0,
                             xl=np.array(lb),
                             xu=np.array(ub), **kwargs)
    
        def _evaluate(self, x, out, *args, **kwargs):   
            if is_MO:
                out["F"] = fit.fun(x.astype(float)) # numba requires all floats
            else:
                out["F"] = fit(x.astype(float))  #  fit returns the score 

    pool = ThreadPool(16)
    problem = MyProblem(runner=pool.starmap, func_eval=starmap_parallelized_eval)
  
    mask = ["int"]*10+["real"]*(dim-10)

    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })
    
    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=0.9, eta=15),
        "int": get_crossover("int_sbx", prob=0.9, eta=15)
    })
    
    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=20),
        "int": get_mutation("int_pm", eta=20)
    })
    
    if is_MO:
        algorithm = NSGA2(
            pop_size=256,
            n_offsprings=10,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        ) 
    else:   
        algorithm = DE(
            pop_size=100,
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
        )
   
    from pymoo.optimize import minimize
        
    res = minimize(problem,
                   algorithm,
                   get_termination("n_gen", 500000),
                   verbose=False)

def check_de_update(dim, fit):
    fit.bounds.lb[:10] = 0 
    fit.bounds.ub[:10] = TRAJECTORY_NUM-1 # integer variables include upper bound 
    # mixed integer multi objective optimization 'modecpp' multi threaded, DE population update
    xs, front = modecpp.retry(fit.fun, fit.nobj, fit.ncon, fit.bounds, num_retries=640, popsize = 128, 
                  max_evaluations = 3000000, nsga_update = False, 
                  logger = logger(), workers=16, 
                  ints=[True]*10+[False]*(dim-10))

def get_fitness():
    name = 'tsin3000.60' # 60 trajectories to choose from
    # name = 'tsin3000.10' # 10 fixed trajectories
    transfers = pd.read_csv('data/' + name + '.xz', sep=' ', usecols=[1,2,3,4,5,6,7], compression='xz',
                    names=['asteroid', 'station', 'trajectory', 'mass', 'dv', 'transfer_start', 'transfer_time'])
    # uncomment to write a clear text csv
    # transfers.to_csv('data/' + name + '.txt', sep=' ', header=False) 

    global TRAJECTORY_NUM, ASTEROID_NUM # adjust number of asteroids / trajectories 
    TRAJECTORY_NUM = int(np.amax(transfers["trajectory"]) + 1)
    ASTEROID_NUM = int(np.amax(transfers["asteroid"]) + 1)
    
    # bounds for the objective function
    dim = 10+2*STATION_NUM-1
    lower_bound = np.zeros(dim)
    # lower_bound[10+STATION_NUM:dim] = 0.00001 
    upper_bound = np.zeros(dim)
    lower_bound[:] = 0.0000001 
    upper_bound[10:] = 0.9999999
    upper_bound[:10] = TRAJECTORY_NUM-0.00001 # trajectory indices
    bounds = Bounds(lower_bound, upper_bound)
    
    fit = fitness(transfers)
    fit.bounds = bounds
    return fit

def optimize():        
    fit = get_fitness()
    
    # check_pymoo(dim, fit, lower_bound, upper_bound, False)
    # check_de_update(dim, fit)
    
    # multi objective optimization 'modecpp' multi threaded, NSGA-II population update
    # xs, front = modecpp.retry(fit.fun, fit.nobj, fit.ncon, fit.bounds, num_retries=640, popsize = 96, 
    #               max_evaluations = 3000000, nsga_update = True, logger = logger(), workers=16)
    
    # smart boundary management (SMB) with DE->CMA
    # store = advretry.Store(fitness(transfers), bounds, num_retries=10000, max_eval_fac=5.0) 
    # advretry.retry(store, de_cma(10000).minimize)    

    # smart boundary management (SMB) with CMA-ES
    # store = advretry.Store(fitness(transfers), bounds, num_retries=10000, max_eval_fac=5.0) 
    # advretry.retry(store, Cma_cpp(10000).minimize)    

    # BiteOpt algorithm multi threaded
    store = retry.Store(fitness(transfers), bounds) 
    retry.retry(store, Bite_cpp(1000000, M=1).minimize, num_retries=3200)    

    # CMA-ES multi threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, Cma_cpp(1000000).minimize, num_retries=3200)    

    # scipy minimize algorithm multi threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, Minimize(1000000).minimize, num_retries=3200)    
    
    # fcmaes differential evolution multi threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, De_cpp(1000000).minimize, num_retries=3200)    

    # scipy differential evolution multi threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, Differential_evolution(1000000).minimize, num_retries=3200) 
    
    # scipy dual annealing multi threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, Dual_annealing(1000000).minimize, num_retries=3200) 
 
    # scipy differential evolution single threaded
    # store = retry.Store(fitness(transfers), bounds) 
    # retry.retry(store, Differential_evolution(1000000).minimize, num_retries=320, workers=1)    

# quality diversity

from fcmaes import diversifier, mapelites

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

def plot_archive(archive, problem):   
    si = archive.argsort()
    ys = archive.get_ys()[si]
    xs = archive.get_xs()[si]
    ds = archive.get_ds()[si]
    ysp = []
    si = np.argsort(ys)
    for i in range(len(si)):
        if ys[i] < np.inf: # throw out invalid
            min_mass = ds[i][0]
            dv_val = ds[i][1]
            score = problem.score(xs[i])          
            ysp.append([min_mass, dv_val, score])
            if score > 5000:
                print(score, min_mass, dv_val)
    ysp = np.array(ysp)
    print(len(ysp))
    plot3d(ysp, "scheduling_nd", 'min mass', 'dv val', 'score')
        
def nd_optimize():
    problem = get_fitness()
    problem.qd_dim = 2
    problem.qd_bounds = Bounds([1.0,15],[2.2,24])
    niche_num = 10000  
    name = "scheduler_nd"
    arch = None
    # mapelites.load_archive('scheduler_nd', problem.bounds, problem.qd_bounds, niche_num) 
    opt_params0 = {'solver':'elites', 'popsize':200}
    opt_params1 = {'solver':'BITE_CPP', 'max_evals':1000000, 'stall_criterion':3}
    arch = diversifier.minimize(
         mapelites.wrapper(problem.qd_fun, 2, interval=100000, save_interval=200000000), 
         problem.bounds, problem.qd_bounds, 
         workers = 32, opt_params=[opt_params0, opt_params1], 
         max_evals=100000000, archive = arch,
         niche_num = niche_num, samples_per_niche = 20)   
    print('final archive:', arch.info())
    arch.save(name)
    plot_archive(arch, problem)

# utility functions

@njit(fastmath=True) 
def next_free(used, p):
    while used[p]:
        p = (p + 1) % used.size
    used[p] = True
    return p

@njit(fastmath=True) 
def disjoined(s, n):
    disjoined_s = np.zeros(s.size, dtype=numba.int32)
    used = np.zeros(n, dtype=numba.boolean)
    for i in range(s.size):
        disjoined_s[i] = next_free(used, s[i])
    return disjoined_s, used

@njit(fastmath=True) 
def timings(x, n):
    times = np.zeros(n+1)
    for i in range(n-1):
        times[i] = MAX_TIME * x[10+STATION_NUM+i]
    times[n-1] = 0
    times[n] = MAX_TIME
    times.sort()
    return times

@njit(fastmath=True) 
def dyson_stations(x, n):
    stations = np.argsort(x[10:10+n])
    # station numbers start with 1
    return np.array([s+1 for s in stations])

@njit(fastmath=True) 
def trajectory_selection(x, n):
    trajectories = np.zeros(10, dtype=numba.int32)
    for i in range(10):
        trajectories[i] = int(x[i])
    return disjoined(trajectories, n)
    
@njit(fastmath=True) 
def score(min_mass, trajectory_dv):
    mass_val = min_mass * 1E-10
    dv_val = 0
    for dv in trajectory_dv:
        dv_val += (1.0 + dv/50.0)**2
    return mass_val / (A_DYSON * A_DYSON * dv_val)

@njit(fastmath=True) 
def score_vals(min_mass, trajectory_dv):
    mass_val = min_mass * 1E-10
    dv_val = 0
    for dv in trajectory_dv:
        dv_val += (1.0 + dv/50.0)**2
    return mass_val / (A_DYSON * A_DYSON * dv_val), dv_val

@njit(fastmath=True)        
def trajectory_dv(asteroid, trajectory, delta_v):
    ast_dv = np.zeros((TRAJECTORY_NUM,ASTEROID_NUM))
    for i in range(asteroid.size):  
        ast_id = int(asteroid[i]) # asteroid transferred
        tr = trajectory[i] # current trajectory
        ast_dv[tr, ast_id] = delta_v[i] # mothership delta velocity to reach the asteroid
    trajectory_dv = np.sum(ast_dv, axis=1)
    return trajectory_dv

@njit(fastmath=True)        
def select_dvs(bdv, x):
    trajectories = trajectory_selection(x, TRAJECTORY_NUM)[0]
    sdv = np.zeros(10)
    for i in range(10):
        sdv[i] = bdv[int(trajectories[i])]
    return sdv

@njit(fastmath=True)        
def ast_num(x, asteroid, trajectory):
    asts = np.zeros((ASTEROID_NUM))
    trajectories = trajectory_selection(x, TRAJECTORY_NUM)[1]
    for i in range(asteroid.size):
        if not trajectories[trajectory[i]]: # trajectory not selected
            continue
        asts[int(asteroid[i])] = 1 # asteroid transferred
    return np.sum(asts)    

def main():
    #optimize()
    nd_optimize()
    
if __name__ == '__main__':
    main()
