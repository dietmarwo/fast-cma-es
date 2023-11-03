'''
GTOC (Global Trajectory Optimization Competition) was initiated 20 years ago by ESA to 
provide a platform for Space Agencies to exchange ideas how to plan space flights. 
Every one or two years the best space trajectory planners in the world, including
JPL (NASA) and ESA, compete solving a complex task. This example is related to GTOC12,
which is about planning the mining of asteroids as efficiently as possible.

Each ship departs from and returns to earth, visiting a number of asteroids in between. 
At the first visit a miner is dropped, at the second visit the mined material is collected.
Each miner collects 10 kg material per year.  
Only one miner per asteroid is allowed, and there are 60000 asteroids to choose from. 
Goal is to maximize the average material mined by each ship. 
The whole mining operation is limited to 15 years (epoch 64328 - epoch 69807 MJD),
see https://gtoc12.tsinghua.edu.cn/competition/theProblem for the original task description. 

This example solves a subtask of GTOC12, 
For a cluster of 44 Asteroids taken from JPLs winning solution
we computed a full asteroid-asteroid transfer graph. This graph
excludes transfers/edges which are too early / too late to facilitate earth->asteroid
or asteroid->earth transfers. 

A possible transfer is defined by the tuple (a1, a2, dm, ep, tof) where:

Graph nodes:
a1: start asteroid of the transfer
a2: target asteroid of the transfer

Graph edge attributes:
dm: delta propellant mass, propellant consumption
ep: start epoch (day)
tof: time of flight, ep + tof is the arrival epoch

For a fixed number of ships, the task is now to maximize the mined material (per ship)
by finding graph paths corresponding to the ship trajectories fulfilling all
constraints:

- A ship may not start heading for the next asteroid before it arrives at an asteroid.
- Propellant consumption (sum of the dm values along the path) is limited (<= 1500 kg) for each ship.
- Each asteroid can be visited at most twice. 
- If a ship arrives at an asteroid which was visited before, it collects 10*y kg material, 
  where y is the time in years since the first visit. 
- If there is no edge to any target asteroid after the arrival time or propellant 
  consumption would exceed the limit the journey for the ship ends. 

This task, although quite similar to ESAs Wormhole task 
https://optimise.esa.int/challenge/spoc-2-wormhole-transportation-network/p/wormhole-transportation-network
differs in several aspects:

- Much lower number of nodes - only 44 instead of 10000. 
- All ships (and not only two) are relevant.
- Number of edges is similar - 246736, but we have for each asteroid 4000-8000 outgoing edges 
  and on average 130 (up to 444) edges between any two asteroids. 
- Any node can be used as start / target node. 

Independent from GTOC12 the subtask shown here represents an interesting variation of the
Multiple Vehicle Routing Problem (https://developers.google.com/optimization/routing/vrp), 
where instead of fixed distances between targets/asteroids there are many different time 
dependent opportunities to fly from one asteroid to another. These opportunities differ
both in time of flight and propellant consumption. Solving this class of problems can
assist the planning of complex space flight operations involving many ships. 
Beside asteroid mining other applications could be space debris removal or maintenance/repair
of large satellite networks.

The task is at the same time quite simple and very hard to solve using search. 
Problem is the "coordination" of dropping the miner and collecting the mined material
between ships. 

If we expand the branches for all ships simultaneously the branching factor explodes. 
But if the ships are processed sequentially it is very hard to come up with a good heuristic
which branches to eliminate. 

Here we use continuous evolutionary optimization - the BiteOpt meta-algorithm which 
dynamically adapts the population update method. 

The objective fitness function looks as follows:

- Let n = 44 be the number of asteroids in the cluster / the number of nodes. 
- X is the argument vector containing 2*n=88 decision variables in the interval [0,1]. 
- p = numpy.argsort(X) is a permutation of the integers between 0 and 88 dependent
  on the argument vector X. 
- seq = [i % n for i in p] is a sequence of 88 asteroid indices where each asteroid occurs exactly twice. 
- remove asteroids from seq, starting from the first one (including its 2nd occurrence) 
  until the asteroid sequence has a specific length l. 
  
We end up with an asteroid sequence of length l depending on the argument vector X which contains
each asteroid exactly twice. l depends on the number of ships n and should be adjusted performing 
test runs of the optimization. l limits the path length for each ship, but if chosen too high 
the optimization itself has to ensure there are only asteroid pairs in all paths. 

Now we follow the graph using the sequence determined by the decision variables as "guidance".
We always choose the first possible transfer between two asteroids/nodes. 
If there is no such transfer - or the ships fuel consumption would be above the limit, we 
"start" the next ship / path. If there is no more ship available we compute the mined material
and use this as objective to maximize. Additional objectives are propellant consumption 
and the sum of the length of all paths added using the
weighted sum approach. Although we are only interested in the mined material, the other
objectives serve as an heuristic "guiding" the optimization.

The result can further be enhanced by applying a second optimization (not include here, 
left as exercise). This time the asteroid sequence is fixed, but instead of the first
possible transfer we choose the nth one, where n is selected according the continuous 
argument vector. If n is in the interval [1, n_max] we convert the continuous argument vector
to integer values using ns = X.astype(int). You need to define a function
'_transfer_n(n, a2, epoch_min, transfers, idx)' selecting the nth instead of the first
possible transfer, and a function 'eval_sequence_n(sequence, ns)' which applies it instead of 
'_transfer(a2, epoch_min, transfers, idx)'. As optimizer parallel BiteOpt can be applied again.

Our chosen propellant limit of 1500kg may be a bit high if you consider material mass and
the propellant needed for the transfers from and to earth. Therefore we need the 
second optimization phase to further reduce propellant consumption. The weighted sum of the
objective function needs to be adapted accordingly. 
'''

import sys
import time

from fcmaes import retry
from fcmaes.optimizer import dtime, Bite_cpp, Cma_cpp, Crfmnes_cpp
from loguru import logger
from numba import njit
import numba
from scipy.optimize import Bounds

import ctypes as ct
import multiprocessing as mp 
import numpy as np 


logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="DEBUG")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="DEBUG")

MIN_MJD = 64328
MAX_MJD = 69807

# Numba is used to dramatically speed up graph traversal and objective function execution
                                       
@njit(fastmath=True, cache=True)
def _mined(asteroids, epochs): # mined material for a sequence of asteroids and epochs
    mined = 0
    for i in range(len(asteroids)-1):
        a1 = asteroids[i]
        for j in range(i+1, len(asteroids)):
            if asteroids[j] == a1:
                mined += abs(epochs[j] - epochs[i])
                break
    return 10*mined/365.25 # mined per year

@njit(fastmath=True, cache=True) # reduce sequence to ast_used asteroids maintaining asteroid pairs
def _reduce(ast_num, ast_used, sequence0): # use first
    used = np.zeros(ast_num, dtype = numba.boolean)
    sequence = np.empty(ast_used, dtype = numba.int32)
    j = 0 # number in sequence
    n = 0 # number of asteroids used
    for i in range(len(sequence0)):
        asteroid = sequence0[i]
        if used[asteroid]:
            sequence[j] = asteroid # always take asteroid already used
            j += 1
        else:
            if n < ast_used: # only take new asteroid if not already ast_used taken
                used[asteroid] = True # mark asteroid as used        
                n += 2
                sequence[j] = asteroid
                j += 1
        if j >= len(sequence):
            return sequence
    return None   
    
@njit(fastmath=True, cache=True)
def _transfer(a2, epoch_min, transfers, idx): # finds the next possible transfer to asteroid a2
    for i in idx: # indices are used to speed up graph traversal
        if transfers[i][2] == a2:
            for transfer in transfers[i+1:]: # numba is essential for performance here
                if transfer[0] < 0: # determines the asteroids
                    return None
                epoch = transfer[1]
                if epoch < epoch_min:
                    continue
                return transfer
    return None
        
class Transfers(object):

    def __init__(self, fname): # load compressed transfer graph
        with np.load(fname + '.npz', allow_pickle=True) as data:
            self.tmap = list(data['tmap']) # transfer graph
            self.idx = list(data['idx']) # indices to speed up graph traversal
            self.ast_to_id = data['ast_to_id'].item() # map to internal asteroid id
            self.id_to_ast = data['id_to_ast'].item() # map to original asteroid id
            self.ast_num = len(self.ast_to_id) # number of asteroids
                
    def get_transfer(self, a1, a2, ep_min): # finds the next possible transfer from asteroid a1 to  a2
        return _transfer(a2, ep_min, self.tmap[a1], self.idx[a1])  

def optimize(fname, max_evals=200000, num_restarts=32, seq=None):
    transfers = Transfers(fname) 
    best_mined = mp.RawValue(ct.c_double, 0) 
    best_y = mp.RawValue(ct.c_double, 1E99) 
    t0 = time.perf_counter()
    evals = mp.RawValue(ct.c_int, 0)  
    ast_num = transfers.ast_num

    ast_used = 68 # corresponds to "l" in the description above. 
    ship_used = 4 # number of ships

    dim = 2*ast_num
    bounds = Bounds([0]*dim, [1-1E-10]*dim) # continous decision variables in the [0,1] interval    
    workers = mp.cpu_count() # we use all CPU cores   
    
    def fit(x): 
        permutation = np.fromiter((a % ast_num for a in np.argsort(x)), dtype = int)
        sequence = _reduce(ast_num, ast_used, permutation) # shorten the sequence of asteroids
        return eval_sequence(sequence)
    
    def eval_sequence(sequence): 
        try:
            ships = 0
            epoch = 0
            mass = 0
            all_mass = []
            y = 0
            epochs = []
            asteroids = []
            start_asts = []
            for i in range(len(sequence)-1):
                a1 = sequence[i]
                a2 = sequence[i+1]   
                # transfer = delta_mass, epoch_start, time_of_flight
                transfer = transfers.get_transfer(a1, a2, epoch)          
                if transfer is None or mass + transfer[0] > 1500: # limit delta mass to 1500 kg
                    ships += 1 # last path ended, start a new ship / path
                    epoch = 0
                    all_mass.append(mass) # accumulate propellant consumption for all paths
                    mass = 0
                    if ships >= ship_used: 
                        break # no ship left
                else:
                    delta_mass, epoch_start, time_of_flight = transfer
                    if epoch == 0: # first asteroid of a path (coming from earth)
                        epochs.append(epoch_start)
                        asteroids.append(a1)
                        start_asts.append(a1)
                    epoch = epoch_start + time_of_flight
                    asteroids.append(a2) # target asteroid a2
                    epochs.append(epoch)
                    mass += delta_mass # accumulate propellant consumption for transfers
            if mass > 0: # increase ast_used if this happens, sequence is too short        
                all_mass.append(mass)                    
            mined = _mined(np.array(asteroids), np.array(epochs)) / ship_used
            y = -mined + 0.1*np.mean(all_mass) - 2*len(asteroids)
            evals.value += 1
            if y < best_y.value or mined > best_mined.value:
                dt = max(1E-9, dtime(t0)) # avoid division by 0           
                am = ",".join(f'{int(m)}' for m in all_mass)
                print(f't = {dt} ev = {evals.value} ev/s = {evals.value/dt:.3f} y = {y:.3f} mined = {mined:.2f} dm = {am} a = {len(asteroids)} {list(sequence)}')
            if y < best_y.value:
                best_y.value = y
            if mined > best_mined.value:
                best_mined.value = mined
            return y
        
        except Exception as ex:
            return 100000        
    
    if seq is None: # optimize using parallel BiteOpt
        retry.minimize_plot("minimize_plot", Bite_cpp(max_evals, M=6), fit, bounds, num_retries=num_restarts, workers=workers, 
        #retry.minimize_plot("minimize_plot", Crfmnes_cpp(2*max_evals, popsize=dim), fit, bounds, num_retries=num_restarts, workers=workers, 
                                statistic_num=5000) 
    else: # if seq is defined we just evaluate it
        return eval_sequence(seq)

def check_solutions():
    seqs = [ # good sample solutions computed using optimize("tmap_jpl1", max_evals=2000000, num_restarts=640)
        [19, 0, 8, 4, 18, 41, 10, 40, 25, 3, 26, 19, 43, 23, 15, 17, 20, 29, 6, 20, 29, 31, 21, 11, 36, 1, 24, 39, 30, 0, 8, 10, 25, 40, 42, 35, 6, 12, 37, 23, 13, 43, 31, 12, 41, 4, 13, 7, 22, 18, 42, 7, 17, 28, 15, 22, 28, 26, 1, 35, 11, 21, 36, 39, 3, 37, 24, 30],
        [22, 30, 13, 0, 19, 38, 20, 42, 11, 31, 29, 36, 2, 40, 17, 32, 16, 2, 27, 35, 37, 41, 18, 17, 32, 37, 13, 8, 41, 10, 28, 16, 11, 20, 29, 31, 36, 39, 14, 4, 8, 6, 35, 27, 39, 24, 18, 14, 42, 7, 5, 28, 10, 40, 25, 24, 38, 43, 0, 19, 5, 25, 7, 30, 22, 43, 4, 6],
        [11, 31, 20, 2, 9, 17, 24, 21, 10, 28, 13, 42, 32, 34, 34, 32, 5, 40, 25, 42, 8, 39, 29, 23, 38, 43, 41, 4, 14, 9, 2, 38, 0, 4, 43, 18, 41, 13, 31, 0, 37, 19, 18, 8, 3, 24, 25, 17, 23, 15, 37, 14, 3, 28, 10, 19, 39, 40, 5, 15, 21, 20, 29, 11, 35, 6, 35, 6],
        ]
    for i, seq in enumerate(seqs): 
        print("solution ", i+1, optimize("tmap_jpl1", seq=seq))
       
if __name__ == '__main__':
    
    check_solutions()
    optimize("tmap_jpl1", max_evals=200000, num_restarts=64)
    # use this if you have a fast many core processor:
    #optimize("tmap_jpl1", max_evals=2000000, num_restarts=640)

