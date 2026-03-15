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
from typing import Any, Callable, Dict, List, TextIO, Union
from scipy.optimize import Bounds
import numpy as np
np.set_printoptions(legacy='1.25') 
import multiprocessing as mp
from multiprocessing import Manager
import ctypes as ct
import json
import sys
from datetime import datetime

# --- Optuna State Codes ---
# 0: RUNNING, 1: COMPLETE, 2: PRUNED, 3: FAIL, 4: WAITING
STATE_COMPLETE = 1
STATE_PRUNED = 2

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
    datetime_start: str 

@dataclass
class Trial_param(Base_message):
    trial_id: int
    param_name: str
    param_value_internal: float
    distribution: Dict[str, Any] 

@dataclass
class Trial_value(Base_message):
    trial_id: int
    state: int
    values: List[float]
    datetime_complete: str 

ObjectiveValue = Union[float, np.ndarray]


def message_to_json(message: Base_message) -> str:
    if isinstance(message, Trial_param):
        data = asdict(message)
        data['distribution'] = json.dumps(data['distribution'])
        return json.dumps(data, separators=(',', ':'))
    else:
        data = asdict(message)
        return json.dumps(data, separators=(',', ':'))
    
def distribution(low: float, high: float) -> Dict[str, Any]:
    distribution_str =  f'{{"name": "FloatDistribution", "attributes": {{"step": null, "low": {low}, "high": {high}, "log": false}}}}'
    return json.loads(distribution_str)

def study_start(worker_id: str, study_name: str, dir: Union[int, List[int]]) -> str:
    msg = Study_start(
        op_code=0, worker_id=worker_id, study_name=study_name, 
        directions=[dir] if np.isscalar(dir) else dir,
    )
    return message_to_json(msg)

def trial_param(worker_id: str,
                trial_id: int,
                param_name: Union[str, int],
                param_value_internal: float,
                low: float,
                high: float) -> str:
    msg = Trial_param(
        op_code=5, worker_id=worker_id, trial_id=trial_id,
        param_name=str(param_name), param_value_internal=param_value_internal,
        distribution=distribution(low, high)
    )
    return message_to_json(msg)

def trial_start(worker_id: str, study_id: int) -> str:
    msg = Trial_start(
        op_code=4, worker_id=worker_id, study_id=study_id, 
        datetime_start=datetime.now().isoformat()
    )
    return message_to_json(msg)

def trial_value(worker_id: str,
                trial_id: int,
                y: ObjectiveValue,
                state: int = STATE_COMPLETE) -> str:
    """
    Creates the JSON message for a trial result.
    Args:
        state (int): 1 for COMPLETE, 2 for PRUNED.
    """
    msg = Trial_value(
        op_code=6, worker_id=worker_id, trial_id=trial_id,
        state=state,
        values=[y] if np.isscalar(y) else y,
        datetime_complete=datetime.now().isoformat()
    )
    return message_to_json(msg)

class Journal:
    def __init__(self, filename: str, study_name: str, dir: Union[int, List[int]]) -> None:
        self.filename = filename
        self.file: TextIO = open(self.filename, 'w')
        self.study("main", study_name, dir)
    
    def study(self, worker_id: str, study_name: str, dir: Union[int, List[int]]) -> None:
        self.file.write(study_start(worker_id, study_name, dir) + '\n')
        self.file.flush()

    def trial(self, worker_id: str, study_id: int) -> None:
        self.file.write(trial_start(worker_id, study_id) + '\n')
        self.file.flush()
    
    def param(self,
              worker_id: str,
              trial_id: int,
              param_name: Union[str, int],
              param_value_internal: float,
              low: float,
              high: float) -> None:
        self.file.write(trial_param(worker_id, trial_id, param_name, param_value_internal, low, high) + '\n')
        self.file.flush() 

    def value(self, worker_id: str, trial_id: int, y: ObjectiveValue, state: int = STATE_COMPLETE) -> None:
        self.file.write(trial_value(worker_id, trial_id, y, state) + '\n')
        self.file.flush() 

class journal_wrapper(object):
    """
    Fitness function wrapper that writes results to an Optuna journal file.
    Automatically marks Inf/NaN results as PRUNED.
    """

    def __init__(self,
                 fit: Callable[[np.ndarray], ObjectiveValue],
                 bounds: Bounds,
                 jfname: str,
                 study_name: str,
                 study_id: int,
                 batch_size: int) -> None:
        self.fit = fit
        self.bounds = bounds
        self.journal = Journal(jfname, study_name, 1) # Direction 1 = minimize
        self.study_id = study_id
        self.batch_size = batch_size
        self.trial_id = mp.RawValue(ct.c_int, 0)
        self.lock = mp.Lock()
        self.mgr = Manager()
        self.reset()

    def reset(self) -> None:
        self.starts = self.mgr.list()  
        self.xs = self.mgr.list()  
        self.ys = self.mgr.list()  

    def store_start(self, worker_id: str, study_id: int) -> None:
        self.starts.append(trial_start(worker_id, study_id) + '\n')
         
    def store_x(self, worker_id: str, trial_id: int, x: np.ndarray) -> None:
        x_str = ''
        for i, xi in enumerate(x):
            x_str += trial_param(worker_id, trial_id, i, xi, self.bounds.lb[i], self.bounds.ub[i]) + '\n'
        self.xs.append(x_str)

    def store_y(self, worker_id: str, trial_id: int, y: ObjectiveValue, state: int) -> None:
        self.ys.append(trial_value(worker_id, trial_id, y, state) + '\n')
    
    def __call__(self, x: np.ndarray) -> ObjectiveValue:
        try:
            # 1. Acquire lock to manage the FIFO buffer logic
            with self.lock:
                n = self.batch_size
                # Flush buffer to file if a batch is complete
                if len(self.ys) >= n:      
                    for i in range(n): self.journal.file.write(self.starts.pop(0))
                    for i in range(n): self.journal.file.write(self.xs.pop(0))
                    for i in range(n): self.journal.file.write(self.ys.pop(0))
                    self.journal.file.flush()
                
                # Get unique trial ID
                trial_id = self.trial_id.value
                self.trial_id.value += 1

            # 2. Prepare metadata
            worker_id = str(trial_id % self.batch_size)
            self.store_start(worker_id, self.study_id)
            self.store_x(worker_id, trial_id, x)
            
            # 3. Calculate Fitness
            y = self.fit(x)
            
            # 4. Determine State (Pruning Logic)
            state = STATE_COMPLETE
            if np.isinf(y) or np.isnan(y):
                state = STATE_PRUNED
            
            # 5. Store result
            self.store_y(worker_id, trial_id, y, state)                
            return y
            
        except Exception as ex:
            print(f"Journal Wrapper Error: {str(ex)}")  
            return sys.float_info.max
