# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""
Simple Optuna Journal file generating wrapper for single and 
multiple objective fcmaes objective functions.
Can be used to gain live insight into a long running optimization process. 

Warning:

- Only use for slow Hyperparameter optimizations, otherwise the journal file will grow too big. 

Usage example:

- https://github.com/dietmarwo/fast-cma-es/blob/master/examples/prophet_opt.py

See

- https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html
- https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/011_journal_storage.html

Usage:

install optuna-dashboard

- pip install optuna-dashboard

optional:

- pip install optuna-fast-fanova gunicorn

Then call:

- optuna-dashboard <path_to_journalfile>

In your browser open:

- http://127.0.0.1:8080/ 

"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from scipy.optimize import Bounds
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
import ctypes as ct
import json
import sys

from datetime import datetime

@dataclass
class Base_message:
    op_code: int
    worker_id: str

@dataclass
class Study_start(Base_message):
    study_name: str
    directions: List[int]

@dataclass
class Trial_start(Base_message):
    study_id: int
    datetime_start: str  # ISO format date-time

@dataclass
class Trial_param(Base_message):
    trial_id: int
    param_name: str
    param_value_internal: float
    distribution: Dict[str, Any]  # Represented as a dictionary

@dataclass
class Trial_value(Base_message):
    trial_id: int
    state: int
    values: List[float]
    datetime_complete: str  # ISO format date-time

def message_to_json(message):
    if isinstance(message, Trial_param):
        data = asdict(message)
        # Serialize 'distribution' field as JSON string
        data['distribution'] = json.dumps(data['distribution'])
        return json.dumps(data, separators=(',', ':'))
    else:
        data = asdict(message)
        return json.dumps(data, separators=(',', ':'))
    
def distribution(low, high):
    distribution_str =  f'{{"name": "FloatDistribution", "attributes": {{"step": null, "low": {low}, "high": {high}, "log": false}}}}'
    return json.loads(distribution_str)

def study_start(worker_id, study_name, dir):
    msg = Study_start(
        op_code=0,
        worker_id=worker_id,
        study_name=study_name, 
        directions=[dir] if np.isscalar(dir) else dir,
    )
    return message_to_json(msg)

def trial_param(worker_id, trial_id, param_name, param_value_internal, low, high):
    msg = Trial_param(
        op_code=5,
        worker_id=worker_id,
        trial_id=trial_id,
        param_name=str(param_name),
        param_value_internal=param_value_internal,
        distribution=distribution(low, high)
    )
    return message_to_json(msg)

def trial_start(worker_id, study_id):
    datetime_str = datetime.now().isoformat()
    msg = Trial_start(
        op_code=4,
        worker_id=worker_id,
        study_id=study_id, 
        datetime_start=datetime_str
    )
    return message_to_json(msg)

def trial_value(worker_id, trial_id, y):
    datetime_str = datetime.now().isoformat()
    msg = Trial_value(
        op_code=6,
        worker_id=worker_id,
        trial_id=trial_id,
        state=1,
        values=[y] if np.isscalar(y) else y,
        datetime_complete=datetime_str
    )
    return message_to_json(msg)

class Journal:
    
    def __init__(self, filename, study_name, dir):
        self.filename = filename
        self.file = open(self.filename, 'w')
        self.study("main", study_name, dir)
    
    def study(self, worker_id, study_name, dir):
        self.file.write(study_start(worker_id, study_name, dir) + '\n')
        self.file.flush()  # Ensure that data is written to disk

    def trial(self, worker_id, study_id):
        self.file.write(trial_start(worker_id, study_id) + '\n')
        self.file.flush()  # Ensure that data is written to disk
    
    def param(self, worker_id, trial_id, param_name, param_value_internal, low, high):
        self.file.write(trial_param(worker_id, trial_id, param_name, param_value_internal, low, high) + '\n')
        self.file.flush()  # Ensure that data is written to disk

    def value(self, worker_id, trial_id, y):
        self.file.write(trial_value(worker_id, trial_id, y) + '\n')
        self.file.flush()  # Ensure that data is written to disk
        
    def write_x(self, worker_id, trial_id, x, bounds):
        for i, xi in enumerate(x):
            self.param(worker_id, trial_id, i, xi, bounds.lb[i], bounds.ub[i])
        
    def write_xs(self, trial_id, xs, bounds):
        for worker_id, x in enumerate(xs):
            self.write_x(str(worker_id+1), trial_id+worker_id, x, bounds)
    
    def write_ys(self, trial_id, ys):
        for worker_id, y in enumerate(ys):
            self.value(str(worker_id+1), trial_id+worker_id, y)

    def write_starts(self, study_id, batch_size):
        for worker_id in range(batch_size):
            self.trial(str(worker_id+1), study_id)

    def close(self):
        self.file.close()    

class journal_wrapper(object):
    """Fitness function wrapper for journal."""

    def __init__(self, fit, bounds, jfname, study_name, study_id, batch_size):
        self.fit = fit
        self.bounds = bounds
        self.journal = Journal(jfname, study_name, 1)
        self.study_id = study_id
        self.batch_size = batch_size
        self.trial_id = mp.RawValue(ct.c_int, 0)
        self.lock = mp.Lock()
        self.mgr = Manager()
        self.reset()

    def reset(self):
        self.starts = self.mgr.list()  
        self.xs = self.mgr.list()  
        self.ys = self.mgr.list()  

    def store_start(self, worker_id, study_id):
        self.starts.append(trial_start(worker_id, study_id) + '\n')
         
    def store_x(self, worker_id, trial_id, x):
        x_str = ''
        for i, xi in enumerate(x):
            x_str += trial_param(worker_id, trial_id, i, xi, self.bounds.lb[i], self.bounds.ub[i]) + '\n'
        self.xs.append(x_str)

    def store_y(self, worker_id, trial_id, y):
        self.ys.append(trial_value(worker_id, trial_id, y) + '\n')
    
    #we need to reorder the journal output to get the dashboard working    
    def __call__(self, x):
        try:
            with self.lock:
                n = self.batch_size
                if len(self.ys) >= n:      
                    for i in range(n):           
                        self.journal.file.write(self.starts.pop(0))
                    for i in range(n): 
                        self.journal.file.write(self.xs.pop(0))
                    for i in range(n): 
                        self.journal.file.write(self.ys.pop(0))
                    self.journal.file.flush()
                trial_id = self.trial_id.value
                self.trial_id.value += 1

            worker_id = str(trial_id % self.batch_size)
            self.store_start(worker_id, self.study_id)
            self.store_x(worker_id, trial_id, x)
            y = self.fit(x)
            self.store_y(worker_id, trial_id, y)                
            return y
        except Exception as ex:
            print(str(ex))  
            return sys.float_info.max  
        