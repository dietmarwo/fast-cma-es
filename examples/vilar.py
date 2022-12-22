'''
Derived from 
https://github.com/StochSS/GillesPy2/blob/main/examples/StartingModels/VilarOscillator/VilarOscillator.py

Optimization driven parameter sweeping

do 'python3 -m pip install gillespy2 --user --upgrade' 
before running this example

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Sweep.adoc for a detailed description.
See also https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Diversity.adoc

'''

import gillespy2
import numpy as np
from scipy.signal import argrelextrema
from fcmaes import mode,  moretry
from scipy.optimize import Bounds
import matplotlib.pyplot as plt    
import multiprocessing as mp
import matplotlib.pyplot as plt

class VilarOscillator(gillespy2.Model):
    def __init__(self, parameter_values=None):
        gillespy2.Model.__init__(self, name="VilarOscillator")
        self.volume = 1

        # Parameters
        alphaA = gillespy2.Parameter(name="alphaA", expression=50)
        alphaA_prime = gillespy2.Parameter(name="alphaA_prime", expression=500)
        alphaR = gillespy2.Parameter(name="alphaR", expression=0.01)
        alphaR_prime = gillespy2.Parameter(name="alphaR_prime", expression=50)
        betaA = gillespy2.Parameter(name="betaA", expression=50)
        betaR = gillespy2.Parameter(name="betaR", expression=5)
        deltaMA = gillespy2.Parameter(name="deltaMA", expression=10)
        deltaMR = gillespy2.Parameter(name="deltaMR", expression=0.5)
        deltaA = gillespy2.Parameter(name="deltaA", expression=1)
        deltaR = gillespy2.Parameter(name="deltaR", expression=0.2)
        gammaA = gillespy2.Parameter(name="gammaA", expression=1)
        gammaR = gillespy2.Parameter(name="gammaR", expression=1)
        gammaC = gillespy2.Parameter(name="gammaC", expression=2)
        thetaA = gillespy2.Parameter(name="thetaA", expression=50)
        thetaR = gillespy2.Parameter(name="thetaR", expression=100)
        
        self.add_parameter([alphaA, alphaA_prime, alphaR, alphaR_prime, betaA, betaR,
                            deltaMA, deltaMR, deltaA, deltaR, gammaA, gammaR, gammaC,
                            thetaA, thetaR])

        # Species
        Da = gillespy2.Species(name="Da", initial_value=1, mode="discrete")
        Da_prime = gillespy2.Species(name="Da_prime", initial_value=0, mode="discrete")
        Ma = gillespy2.Species(name="Ma", initial_value=0, mode="discrete")
        Dr = gillespy2.Species(name="Dr", initial_value=1, mode="discrete")
        Dr_prime = gillespy2.Species(name="Dr_prime", initial_value=0, mode="discrete")
        Mr = gillespy2.Species(name="Mr", initial_value=0, mode="discrete")
        C = gillespy2.Species(name="C", initial_value=0, mode="discrete")
        A = gillespy2.Species(name="A", initial_value=0, mode="discrete")
        R = gillespy2.Species(name="R", initial_value=0, mode="discrete")
        
        self.add_species([Da, Da_prime, Ma, Dr, Dr_prime, Mr, C, A, R])

        # Reactions
        r1 = gillespy2.Reaction(name="r1", reactants={'A': 1, 'R': 1}, products={'C': 1}, rate="gammaC")
        r2 = gillespy2.Reaction(name="r2", reactants={'A': 1}, products={}, rate="deltaA")
        r3 = gillespy2.Reaction(name="r3", reactants={'C': 1}, products={'R': 1}, rate="deltaA")
        r4 = gillespy2.Reaction(name="r4", reactants={'R': 1}, products={}, rate="deltaR")
        r5 = gillespy2.Reaction(name="r5", reactants={'A': 1, 'Da': 1}, products={'Da_prime': 1}, rate="gammaA")
        r6 = gillespy2.Reaction(name="r6", reactants={'Da_prime': 1}, products={'A': 1, 'Da': 1}, rate="thetaA")
        r7 = gillespy2.Reaction(name="r7", reactants={'Da': 1}, products={'Da': 1, 'Ma': 1}, rate="alphaA")
        r8 = gillespy2.Reaction(name="r8", reactants={'Da_prime': 1}, products={'Da_prime': 1, 'Ma': 1}, rate="alphaA_prime")
        r9 = gillespy2.Reaction(name="r9", reactants={'Ma': 1}, products={}, rate="deltaMA")
        r10 = gillespy2.Reaction(name="r10", reactants={'Ma': 1}, products={'A': 1, 'Ma': 1}, rate="betaA")
        r11 = gillespy2.Reaction(name="r11", reactants={'A': 1, 'Dr': 1}, products={'Dr_prime': 1}, rate="gammaR")
        r12 = gillespy2.Reaction(name="r12", reactants={'Dr_prime': 1}, products={'A': 1, 'Dr': 1}, rate="thetaR")
        r13 = gillespy2.Reaction(name="r13", reactants={'Dr': 1}, products={'Dr': 1, 'Mr': 1}, rate="alphaR")
        r14 = gillespy2.Reaction(name="r14", reactants={'Dr_prime': 1}, products={'Dr_prime': 1, 'Mr': 1}, rate="alphaR_prime")
        r15 = gillespy2.Reaction(name="r15", reactants={'Mr': 1}, products={}, rate="deltaMR")
        r16 = gillespy2.Reaction(name="r16", reactants={'Mr': 1}, products={'Mr': 1, 'R': 1}, rate="betaR")
        
        self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r8, r9,
                           r10, r11, r12, r13, r14, r15, r16])

        # Timespan
        self.timespan(np.linspace(0,400,401))

def get_bounds(model, scale):
    lower = []
    upper = []
    for _, param in model.listOfParameters.items():
        lower.append(param.value/scale)
        upper.append(param.value*scale)    
    return Bounds(lower, upper)

def set_params(model, x):
    for i, (_, param) in enumerate(model.listOfParameters.items()):
        param.value = x[i]
    
def sweep_params():
    
    # multi processing result list
    results = mp.Manager().list() 
    
    class fcmaes_problem():
         
        def __init__(self):
            self.bounds = get_bounds(VilarOscillator(), 100)
            self.dim = len(self.bounds.ub)
             
        def fitness(self, x):
            model = VilarOscillator()
            set_params(model, x)
            res = model.run(algorithm = "SSA")
            # store params, result tuple
            results.append((x, res))
            R = res['R'] # time series for R
            r_mean = np.mean(R)
            r_over = np.array(np.fromiter((r for r in R if r > r_mean), dtype=float))
            ilocs_max = argrelextrema(r_over, np.greater_equal, order=3)[0]
            freq = len(ilocs_max) / len(R)
            peak_dists = np.array(np.fromiter((ilocs_max[i] - ilocs_max[i-1] for i in range(1, len(ilocs_max))), dtype=float))
            sdev_peak_dist = np.std(peak_dists)
            peaks = (r_over - r_mean)[ilocs_max]
            sdev_amp = np.std(peaks)
            # maximize sdev_peak_dist and sdev_amp
            return [-sdev_peak_dist, -sdev_amp, freq]
            
    problem = fcmaes_problem()   
    popsize = 64 # population size of the evolutionary algorithm
    max_evaluations = popsize*16 # maximum number of evaluation
    # popsize = 256 # population size of the evolutionary algorithm
    # max_evaluations = popsize*96 # maximum number of evaluation
    nobj = 3 # number of objectives
    ncon = 0 # number of constraints
    # stores all values; if capacity is reached, content is replaced by the pareto front 
    store = mode.store(problem.dim, nobj, capacity=max_evaluations) 
    # perform the parameter sweep by multi objective optimization
    xs, ys = mode.minimize(mode.wrapper(problem.fitness, 3, interval=64, store=store), 
                                    nobj, ncon,
                                    problem.bounds, popsize = popsize, 
                                    max_evaluations = max_evaluations, 
                                    nsga_update=False, workers=min(popsize, mp.cpu_count()))
    # save last population of the evolutionary algorithm
    np.savez_compressed("sweep", xs=xs, ys=ys)     
    # save all evaluations
    xs, ys = store.get_xs(), store.get_ys()
    np.savez_compressed("sweep_all", xs=xs, ys=ys)   
    # show results
    for x, res in results[:10]:
        print(list(x), list(res['R']))
    moretry.plot("sweep_all", ncon, xs, ys) # plot 2d
    plot3d(xs, ys, "sweep_3d") # plot 3d

from fcmaes import diversifier, mapelites
import threadpoolctl
from elitescass2 import plot3d

def run_diversifier():
        
    class nd_problem():
     
        def __init__(self):
            self.bounds = get_bounds(VilarOscillator(), 100)
            self.qd_bounds = Bounds([0, 30, .035], [3, 300, .050])
            self.qd_dim = 3
            self.dim = len(self.bounds.ub)
             
        def fitness(self, x):
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                model = VilarOscillator()
                set_params(model, x)
                res = model.run(algorithm = "SSA")
                R = res['R'] # time series for R
                r_mean = np.mean(R)
                r_over = np.array(np.fromiter((r for r in R if r > r_mean), dtype=float))
                ilocs_max = argrelextrema(r_over, np.greater_equal, order=3)[0]
                freq = len(ilocs_max) / len(R)
                peak_dists = np.array(np.fromiter((ilocs_max[i] - ilocs_max[i-1] for i in range(1, len(ilocs_max))), dtype=float))
                sdev_peak_dist = np.std(peak_dists)
                peaks = (r_over - r_mean)[ilocs_max]
                sdev_amp = np.std(peaks)
                ws = sdev_peak_dist/3.0 + sdev_amp/300.0 # weighted sum               
                #ws = 2 - (sdev_peak_dist/3.0 + sdev_amp/300.0) # maximize sdev               
                return ws, np.array([sdev_peak_dist, sdev_amp, freq])
           
    problem = nd_problem() 

    opt_params0 = {'solver':'elites', 'popsize':8}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':200, 'popsize':16, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem.fitness, problem.qd_dim, interval=100, save_interval=4000), 
         problem.bounds, problem.qd_bounds, opt_params=[opt_params0, opt_params1], max_evals=12800)
    print("final archive: " + archive.info())
    archive.save("vilar_qd")
    plot_archive(archive)
    
def plot_archive(archive):
    cs = archive.get_cs()
    ds = archive.get_ds()
    
    si = archive.argsort()
    ysp = []
    descriptions = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    for i in range(len(si)):
        desc = descriptions[i]
        ysp.append([desc[0], desc[1], desc[2]])

    ysp = np.array(ysp)
    print(len(ysp))
    print(ysp)
    plot3d(ysp, "vilar_qd", 'sdev peak distance', 'sdev amplitude', 'frequency')

if __name__ == '__main__':
    #sweep_params()
    run_diversifier()
    #plot()
    
    
