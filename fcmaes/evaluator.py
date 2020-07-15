# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Parallel objective function evaluator.
    Uses pipes to avoid re-spawning new processes for each eval_parallel call. 
    the objective function is distributed once to all processes and
    reused for all eval_parallel calls. Evaluator(fun) needs to be stopped after the
    whole optimization is finished to avoid a resource leak.
"""

from multiprocessing import Process, Pipe
import multiprocessing as mp
import numpy as np
import sys 
import math   

def eval_parallel(xs, evaluator):
    popsize = len(xs)
    ys = np.empty(popsize)
    pipe_limit = 256
    i0 = 0
    i1 = min(popsize, pipe_limit)
    while True:
        _eval_parallel_segment(xs, ys, i0, i1, evaluator)
        if i1 >= popsize:
            break;
        i0 += pipe_limit
        i1 = min(popsize, i1 + pipe_limit)
    return ys

class Evaluator(object):
       
    def __init__(self, 
                 fun, # objective function
                ):   
        self.fun = fun 
        self.pipe = Pipe()
        self.read_mutex = mp.Lock() 
        self.write_mutex = mp.Lock() 
            
    def start(self, workers=mp.cpu_count()):
        self.workers = workers
        self.proc=[Process(target=_evaluate, args=(self.fun, 
                self.pipe, self.read_mutex, self.write_mutex)) for _ in range(workers)]
        [p.start() for p in self.proc]
        
    def stop(self): # shutdown all workers 
        for _ in range(self.workers):
            self.pipe[0].send(None)
        [p.join() for p in self.proc]    
        for p in self.pipe:
            p.close()

def _eval_parallel_segment(xs, ys, i0, i1, evaluator):
    for i in range(i0, i1):
        evaluator.pipe[0].send((i, xs[i]))
    for _ in range(i0, i1):        
        i, y = evaluator.pipe[0].recv()
        ys[i] = y
    return ys

def _evaluate(fun, pipe, read_mutex, write_mutex): # worker
    while True:
        with read_mutex:
            msg = pipe[1].recv() # Read from the input pipe
        if msg is None: 
            break # shutdown worker
        try:
            i, x = msg
            y = fun(x)
            if not math.isfinite(y):
                y = sys.float_info.max
        except Exception:
            y =  sys.float_info.max
        with write_mutex:            
            pipe[1].send((i, y)) # Send result
