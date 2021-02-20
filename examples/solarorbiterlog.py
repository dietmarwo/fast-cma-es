'''
Created on Jan 10, 2020

@author: Dietmar Wolz
'''

# Analyzes the logs produced by solarorbitermulti.py in folder log.
# Produces a sorted list of list each representing the best solution 
# for a sequence found in the logs. The best sequence is listed first.   

import glob
import numpy as np
import time
import pygmo as pg
from pykep import AU, epoch
from pykep.planet import jpl_lp
from pykep.trajopt.gym._solar_orbiter import _solar_orbiter_udp

tmin = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 - 2*365)
tmax = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 + 2*365)

def value(x, seq):
    solar_orbiter = _solar_orbiter_udp([tmin, tmax], seq=seq)
    prob = pg.problem(pg.unconstrain(solar_orbiter,method="weighted",weights=[1.0, 10.0, 100, 100]))
    return prob.fitness(x)[0] 

def read_csv(fname):
    data = []
    files = glob.glob(fname, recursive=True)
    
    for fname in files:
        with open(fname) as csvfile:
            lines = csvfile.readlines()
            for line in lines:
                try:
                    row = line.split(' ')
                    if len(row) < 11:
                        continue
                    if row[0] == "problem":
                        seqs = line[line.rfind(']')+2:-1]
                        seq = seqs.split(' ')
                        seq = [jpl_lp(p) for p in seq]
                        continue
                    xs = line[line.rfind('[')+1:line.rfind(']')]
                    x = xs.split(', ')
                    x = [float(xi) for xi in x]
                    y = value(x, seq)
                    data.append([y,x,seq,seqs])
                except Exception as ex:
                    pass
    return data

def check_log(fname):
    try:
        data = read_csv(fname)
        val_map = {}
        for y, x, seq, seqs in data:
            if seqs in val_map:
                y_, x_, _ = val_map[seqs]
                if y < y_:
                    val_map[seqs] = [y, x, seqs]
            else:
                val_map[seqs] = [y, x, seqs]
        values = np.array(list(val_map.values()))
        ys = np.array([v[0] for v in values])
        idx = ys.argsort()
        sorted = values[idx]
        for v in sorted:
            if v[0] < 3.0:
                seq = '[' + v[2].replace(' ', ', ') + ']'
                x = str(v[1])
                y = str(v[0])
                print('\t' + '[ ' + y + ', ' + seq + ', ' + x + ' ],')         
    except Exception as ex:
        print(ex)
        

if __name__ == '__main__':
    check_log('logs/*')
#    check_log('logs/len10.log')
    
    
