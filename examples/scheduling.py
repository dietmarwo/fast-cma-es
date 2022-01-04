# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import math
import pandas as pd
import numpy as np
import sys, math, time
from fcmaes import retry, advretry
from fcmaes.optimizer import logger, Bite_cpp, De_cpp, de_cma, dtime, Differential_evolution
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
from numba import njit

STATION_NUM = 12 # number of dyson ring stations
TRAJECTORY_NUM = 50 # we select 10 mothership trajectories from these trajectories

MAX_TIME = 20.0 # mission time in years
ASTEROID_NUM = 83454 # number of asteroids
WAIT_TIME = 90/365.25 # years, after arrival wait until construction may start
ALPHA = 6.0e-9 # conversion factor time of flight -> arrival mass
A_DYSON = 1.3 # size of the dyson ring in AU

DAY = 24 * 3600
YEAR = DAY*365.25

@njit(fastmath=True)        
def select(asteroid, station, trajectory, mass, transfer_start, transfer_time, x):
    trajectories = trajectory_selection(x, TRAJECTORY_NUM)[1] # select 10 trajectories representing the 10 mothership trajectories
    stations = dyson_stations(x, STATION_NUM) # derive dyson_stations targeted at time slots from argument vector
    times = timings(x, STATION_NUM) # derive station build time slot boundaries from argument vector (in years)
    slot_mass = np.zeros(STATION_NUM) # mass sum per time slot
    ast_val = np.zeros(ASTEROID_NUM) # deployed mass for asteroid
    ast_slot = np.zeros(ASTEROID_NUM) # build time slot used for asteroid
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
                        tof += min_time - arrival_time # tof increases
                    mval = (1.0 - YEAR*tof*ALPHA) * m # estimated asteroid mass at arrival time 
                    if ast_val[ast_id] > 0: # asteroid already transferred
                        if ast_val[ast_id] < mval: # replace with actual transfer, remove old asteroid mass
                            slot_mass[int(ast_slot[ast_id])] -= ast_val[ast_id]
                        else: # keep old transfer, don't use the new one
                            mval = 0;
                    if mval > 0:  # register actual transfer
                        slot_mass[slot] += mval
                        ast_val[ast_id] = mval
                        ast_slot[ast_id] = slot                  
    slot_mass.sort()
    min_mass = slot_mass[0]
    f = 1.0
    for m in slot_mass:
        min_mass += f*m
        f *= 0.5
    return min_mass, slot_mass

class fitness(object): # the objective function

    def __init__(self, transfers):
        self.evals = mp.RawValue(ct.c_long, 0)  # writable across python processes
        self.best_y = mp.RawValue(ct.c_double, math.inf) # writable across python processes
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
        
    def __call__(self, x):     
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

def optimize():    
    transfers = pd.read_csv('data/asteroid_transfers1.3.xz', sep=' ', usecols=[1,2,3,4,5,6,7], compression='xz',
                    names=['asteroid', 'station', 'trajectory', 'mass', 'dv', 'transfer_start', 'transfer_time'])
    # uncomment to write a clear text csv
    # transfers.to_csv('asteroid_transfers1.3.txt', sep=' ', header=False) 

    # bounds for the objective function
    dim = 10+2*STATION_NUM-1
    lower_bound = np.zeros(dim)
    lower_bound[10+STATION_NUM:dim] = 0.001 
    upper_bound = np.zeros(dim)
    upper_bound[:10] = TRAJECTORY_NUM # trajectory indices
    upper_bound[10:10+STATION_NUM] = STATION_NUM-0.00001 # station indices, avoid rounding errors
    upper_bound[10+STATION_NUM:dim] = 0.999 # Dyson station build time windows
    bounds = Bounds(lower_bound, upper_bound)
    
    store = retry.Store(fitness(transfers), bounds, logger=logger()) 
    
    # use BiteOpt algorithm with parallel retry
    retry.retry(store, Bite_cpp(1000000, M=16).minimize, num_retries=320)    
    
    # uncomment to try scipy differential evolution single threaded
    # retry.retry(store, Differential_evolution(1000000).minimize, num_retries=320, workers=1)    
   
    # uncomment to try smart boundary management (SMB)
    # store = advretry.Store(fitness(transfers), bounds, num_retries=100000, logger=logger())
    # advretry.retry(store, de_cma(1500).minimize) 
      
    return store.get_xs(), store.get_ys()

# utility functions

@njit(fastmath=True) 
def next_free(used, p):
    p = int(p)
    while used[p] > 0:
        p = (p + 1) % used.size
    used[p] = 1
    return p

@njit(fastmath=True) 
def disjoined(s, n):
    disjoined_s = np.zeros(s.size)
    used = np.zeros(n)
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
    stations = np.zeros(n)
    for i in range(n):
        stations[i] = int(x[10+i])
    # station numbers start with 1
    return np.array([s+1 for s in disjoined(stations, n)[0]])

@njit(fastmath=True) 
def trajectory_selection(x, n):
    trajectories = np.zeros(10)
    for i in range(10):
        trajectories[i] = int(x[i])
    return disjoined(trajectories, n)
    
@njit(fastmath=True) 
def score(min_mass, trajectory_dv):
    mass_val = min_mass * 1E-10
    dv_val = 0
    for dv in trajectory_dv:
        dv_val += (1.0 + dv/50.0)**2
    return mass_val / (A_DYSON * A_DYSON * dv_val);

@njit(fastmath=True)        
def trajectory_dv(asteroid, trajectory, delta_v):
    ast_dv = np.zeros((TRAJECTORY_NUM,ASTEROID_NUM))
    for i in range(asteroid.size):  
        ast_id = int(asteroid[i]) # asteroid transferred
        tr = int(trajectory[i]) # current trajectory
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
        if trajectories[int(trajectory[i])] == 0: # trajectory not selected
            continue
        asts[int(asteroid[i])] = 1 # asteroid transferred
    return np.sum(asts)    

def main():
   optimize()
    
if __name__ == '__main__':
    main()