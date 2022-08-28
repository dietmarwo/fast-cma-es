''' 
Tests performance of CR-FM-NES and SPSA using parallel optimization retry for the quantum
implementation of maxcut by optimizing the parameters of a 
VQE (variational quantum eigensolver)

This code is adapted from
https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html

Changes:

qiskit was largely refactored lately, a lot of imports needed to be changed

Added two optimizer wrappers:
- wrapped SPSA as fcmaes optimizer (fcmaes_SPSA)
- wrapped fcmaes optimizers + parallel retry as qiskit optimizer
 
These packages need to be installed before executing the code:
pip install quiskit
pip install quiskit-optimization
pip install fcmaes

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Quant.adoc for a detailed description.
'''

import numpy as np
import networkx as nx

from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo
from qiskit.algorithms import optimizers
from qiskit.algorithms.optimizers import OptimizerSupportLevel, OptimizerResult
from qiskit.algorithms.optimizers.optimizer import POINT

from typing import Optional, Callable, List, Dict, Any, Tuple
from scipy.optimize import Bounds
import threadpoolctl
import multiprocessing as mp
    
from fcmaes.optimizer import Optimizer, Crfmnes_cpp, Crfmnes, Cma_python, wrapper, logger 
from fcmaes import retry

# wraps SPSA as fcmaes optimizer
class fcmaes_SPSA(Optimizer):

    def __init__(self, maxiter=1000):        
        Optimizer.__init__(self, maxiter, 'SPSA')
        self.opt = SPSA(maxiter=maxiter) # guessing

    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=None, store=None):
        if guess is None: # necessary for parallel retry
            guess = np.random.uniform(bounds.lb, bounds.ub) if rg is None else \
                    rg.uniform(bounds.lb, bounds.ub)    
        ret = self.opt.minimize(fun, guess, bounds=[t for t in zip(bounds.lb, bounds.ub)])
        return ret.x, ret.fun, ret.nfev

# wraps fcmaes optimizers + parallel retry as qiskit optimizer
class fcmaes_Optimizer(optimizers.Optimizer):

    def __init__(
        self,
        optimizer, # used qiskit optimizer
        bounds: Optional[Bounds] = None, # variable bounds
        max_retries: int = 1, # number of parallel optimization retries
        workers: int = None, # maximal number of parallel workers, if None uses all physical available threads
        use_wrapper = False, # monitors optimization progress
        logger = None # monitors parallel retry
        
    ) -> None:
        self._optimizer = optimizer
        self._bounds = bounds
        self._max_retries = max_retries
        self._workers = mp.cpu_count() if workers is None else workers 
        self._use_wrapper = use_wrapper
        self._logger = logger
        
    def get_support_level(self):
        """Returns support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.required,
            "initial_point": OptimizerSupportLevel.supported,
        }

    @property
    def settings(self) -> Dict[str, Any]:
        return {"optimizer": self._optimizer.name, "_max_retries": self._max_retries}

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        bnds = self._bounds if bounds is None else Bounds([b[0] for b in bounds], [b[1] for b in bounds])
        if self._use_wrapper:
            fun = wrapper(fun) # monitors progress for all parallel processes
        result = OptimizerResult()
        if self._max_retries <= 1:
            x, y, nfev = self._optimizer.minimize(fun, bnds, guess=x0)
            result.x = x
            result.fun = y
            result.nfev = nfev
        else: 
            ret = retry.minimize(fun, bnds, num_retries=self._max_retries, workers = self._workers,
                              optimizer=self._optimizer, logger=self._logger)
            result.x = ret.x
            result.fun = ret.fun
            result.nfev = ret.nfev
        return result

def maxcut(optimizer, # used quiskit optimizer
           n = 13, # number of nodes in graph
           backend_name = "aer_simulator", 
           add_noise=False, # add noise to simulator
           ):
    G = nx.dense_gnm_random_graph(n, 2*n, seed=123)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = 1
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    print(w)
    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    print(qp.prettyprint())
    
    qubitOp, offset = qp.to_ising()
    print("Offset:", offset)
    print("Using Hamiltonian:")
    print(str(qubitOp))
    
    ee = NumPyMinimumEigensolver()    
    result = ee.compute_minimum_eigenvalue(qubitOp)
    x = max_cut.sample_most_likely(result.eigenstate)
    print("energy:", result.eigenvalue.real)
    print("max-cut objective:", result.eigenvalue.real + offset)
    print("solution:", x)
    print("solution objective:", qp.objective.evaluate(x))
    
    backend = Aer.get_backend(backend_name)

    if add_noise:    
        device_backend = FakeVigo()
        device = QasmSimulator.from_backend(device_backend)
        noise_model = NoiseModel.from_backend(device)
        print(noise_model)
        quantum_instance = QuantumInstance(backend, noise_model=noise_model)
    else:
        quantum_instance = QuantumInstance(backend)
    
    ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    vqe = VQE(ry, optimizer=optimizer, quantum_instance=quantum_instance)    
    dim = vqe.ansatz.num_parameters
    vqe.ansatz.parameter_bounds = [(-2 * np.pi, 2 * np.pi)]*dim
    
    # run VQE
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"): 
        # blas threading restriction, speeds up "aer_simulator_statevector" 
        result = vqe.compute_minimum_eigenvalue(qubitOp)
    
    # print results
    x = max_cut.sample_most_likely(result.eigenstate)
    print("energy:", result.eigenvalue.real)
    print("time:", result.optimizer_time)
    print("max-cut objective:", result.eigenvalue.real + offset)
    print("solution:", x)
    print("solution objective:", qp.objective.evaluate(x))

if __name__ == '__main__':
    n = 13 # number of nodes in graph
    
    # maxcut(SPSA(maxiter=1000), n, "aer_simulator") 
    # maxcut(fcmaes_Optimizer(Crfmnes_cpp(2000, popsize=7), use_wrapper=True), n, "aer_simulator")
    
    maxcut(fcmaes_Optimizer(Crfmnes_cpp(8000, popsize=16, workers=16), use_wrapper=True), n, "aer_simulator")
    #maxcut(fcmaes_Optimizer(Crfmnes(8000, popsize=16, workers=16), use_wrapper=True), n, "aer_simulator")
    #maxcut(fcmaes_Optimizer(Cma_python(8000, popsize=16, workers=16), use_wrapper=True), n, "aer_simulator")
    
    # maxcut(fcmaes_Optimizer(fcmaes_SPSA(1000), max_retries=16, use_wrapper=True, logger=logger()), n, "aer_simulator")    
    # maxcut(fcmaes_Optimizer(Crfmnes_cpp(2000, popsize=7), max_retries=16, use_wrapper=True, logger=logger()), n, "aer_simulator")
    # maxcut(fcmaes_Optimizer(Crfmnes_cpp(2000, popsize=7), use_wrapper=True), n, "aer_simulator", add_noise=True)
    
    # maxcut(fcmaes_Optimizer(Crfmnes_cpp(2000, popsize=7), max_retries=8, use_wrapper=True, logger=logger()), n, "aer_simulator", add_noise=True)
    # maxcut(fcmaes_Optimizer(fcmaes_SPSA(1000), max_retries=8, use_wrapper=True, logger=logger()), n, "aer_simulator", add_noise=True)
