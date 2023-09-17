'''
Adapted from "Example with a Single Qubit Variational Form"

https://qiskit.org/textbook/ch-applications/vqe-molecules.html#Example-with-a-Single-Qubit-Variational-Form

Read https://qiskit.org/documentation/getting_started.html about setting up your environment

Install dependencies:
pip install qiskit
pip install qiskit-aer-gpu (doesn't work on AMD GPUs and is not required for executing this file)

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Quant.adoc for a detailed description.
'''
# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

from qiskit import QuantumCircuit, assemble, Aer, transpile, ClassicalRegister, QuantumRegister
from qiskit.algorithms.optimizers import COBYLA
import numpy as np
from time import perf_counter
from fcmaes.optimizer import Bite_cpp, dtime, wrapper
from scipy.optimize import Bounds
from fcmaes import retry, de, cmaes, bitecpp, cmaescpp, decpp

backend = Aer.get_backend("qasm_simulator", max_parallel_threads=1)
#backend.set_options(device='GPU') # if you switch GPU on, parallel simulation will crash

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

NUM_SHOTS = 10000

def get_var_form(params):
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr[0])
    return qc

def get_probability_distribution(counts):
    output_distr = [v / NUM_SHOTS for v in counts.values()]
    if len(output_distr) == 1:
        output_distr.append(1 - output_distr[0])
    return output_distr

def random_target_distr():
    target_distr = np.random.rand(2)
    # We now convert the random vector into a valid probability vector
    target_distr /= sum(target_distr)
    return target_distr

def objective_function(params, target_distr):
    # Obtain a quantum circuit instance from the parameters
    qc = get_var_form(params)
    # Execute the quantum circuit to obtain the probability distribution associated with the current parameters
    t_qc = transpile(qc, backend)
    qobj = assemble(t_qc, shots=NUM_SHOTS)
    result = backend.run(qobj).result()
    # Obtain the counts for each measured state, and convert those counts into a probability vector
    output_distr = get_probability_distribution(result.get_counts(qc))
    # Calculate the cost as the distance between the output distribution and the target distribution
    cost = sum([np.abs(output_distr[i] - target_distr[i]) for i in range(2)])
    return cost

class Fitness(object):
    
    def __init__(self, target_distr):
        self.target_distr = target_distr
        self.bounds = Bounds([0]*3, [2]*3)      
        
    def __call__(self, x):
        try:     
            y = objective_function(x, self.target_distr)
            if not np.isfinite(y) or y > 1E9: 
                return 1E9
            else:
                return y
        except Exception as ex:
            print(str(ex))     
            return 1E9
    
def opt_differential_evolution_loop(fits):
    t0 = perf_counter()
    distances = []
    for fit in fits: 
        ret = de.minimize(fit, 3, fit.bounds, max_evaluations = 1000, 
                          stop_fitness = 0.00001, workers=16)
        print("de time", dtime(t0), "distance", ret.fun)
        distances.append(ret.fun)
    print("de mean distance = " +  str(np.mean(distances)))
    print("de std distance = " +  str(np.std(distances)))

def opt_cmaes_loop(fits):
    t0 = perf_counter()
    distances = []
    for fit in fits: 
        ret = cmaes.minimize(fit, fit.bounds, input_sigma=0.7, max_evaluations = 1000, 
                            stop_fitness = 0.00001, workers=16)
        print("cmaes time", dtime(t0), "distance", ret.fun)
        distances.append(ret.fun)
    print("cmaes mean distance = " +  str(np.mean(distances)))
    print("cmaes std distance = " +  str(np.std(distances)))
 
def opt_biteopt_loop(fits):
    t0 = perf_counter()
    distances = []
    for fit in fits:  
        ret = retry.minimize(fit, fit.bounds,
                              num_retries=16, optimizer=Bite_cpp(100), workers=16)
        print("bite time", dtime(t0), "distance", ret.fun)
        distances.append(ret.fun)
    print("bite mean distance = " +  str(np.mean(distances)))
    print("bite std distance = " +  str(np.std(distances)))

def opt_COBYLA_evolution_loop(fits):
    # Initialize the COBYLA optimizer
    # Create the initial parameters (noting that our single qubit variational form has 3 parameters)
    t0 = perf_counter()
    distances = []
    for fit in fits:
        params = np.random.rand(3)
        optimizer = COBYLA(maxiter=50000, tol=0.00001)
        ret = optimizer.minimize(fun=fit, x0=params)
        print("COBYLA time", dtime(t0), "distance", ret.fun)
        distances.append(ret.fun)
    print("COBYLA mean distance = " +  str(np.mean(distances)))
    print("COBYLA std distance = " +  str(np.std(distances)))
    
def find_COBYLA_weakness():
    
    def fitness(x):
        params = x[:3] # use first three decision variables as guess for COBYLA
        target_distr =  x[3:] # use two decision variables as target
        if min(target_distr) < 0.01: return 0 # avoid division by 0
        target_distr /= sum(target_distr)        
        fit = Fitness(target_distr)

        ret = COBYLA(maxiter=50000, tol=0.00001).minimize(fun=fit, x0=params)
        #ret = bitecpp.minimize(fit, fit.bounds, x0=params, max_evaluations=1000, M=6)
        #ret = bitecpp.minimize(fit, fit.bounds, x0=params, max_evaluations=300, stop_fitness=0.05)
        #ret = cmaescpp.minimize(fit, fit.bounds, x0=params, max_evaluations=300, stop_fitness=0.05)
        #ret = decpp.minimize(fit, 3, fit.bounds, max_evaluations=300, stop_fitness=0.05)
        
        return -ret.fun # we maximize the distance
        
    bounds = Bounds([0]*5, [2]*3 + [1]*2)
    ret = de.minimize(wrapper(fitness), 5, bounds, max_evaluations = 300, workers=16)
    print("worst COBYLA distance = " +  str(ret.fun))
 
if __name__ == '__main__':
    
    # generate Fitness objects associated to random target distributions
    fits = [Fitness(random_target_distr()) for i in range(10)]
    opt_differential_evolution_loop(fits)
    opt_cmaes_loop(fits)
    opt_biteopt_loop(fits)
    opt_COBYLA_evolution_loop(fits)
    # find_COBYLA_weakness() # check for weaknesses of COBYLA
    pass