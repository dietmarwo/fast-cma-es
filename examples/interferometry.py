# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and is extended here by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# Tested with Anaconda 2020.11 https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# Corresponds to the equivalent python example
# https://github.com/dietmarwo/fcmaes-java/blob/master/src/main/java/fcmaes/examples/Interferometry.java
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

import pygmo as pg
from time import time
from interferometryudp import Interferometry
from fcmaes import de, cmaes, retry, advretry
from fcmaes.optimizer import single_objective, de_cma_py, Cma_python, De_python, Cma_cpp, De_cpp, de_cma

udp = Interferometry(11, './img/orion.jpg', 512) 
#udp = Interferometry(5, './img/orion.jpg', 32)

def archipelago():    
    print('interferometer sga archipelago')
    uda = pg.sga(gen = 50000)
    # instantiate an unconnected archipelago
    archi = pg.archipelago(t = pg.topologies.unconnected())
    t = time()
    for _ in range(8):
        alg = pg.algorithm(uda)
        #alg.set_verbosity(1)    
        prob = pg.problem(udp)
        pop = pg.population(prob, 20)    
        isl = pg.island(algo=alg, pop=pop)
        archi.push_back(isl)   
    
    archi.evolve()
    archi.wait_check()
    print(f'archi: {time() - t:0.3f}s')

def check_good_solution():
    fprob = single_objective(pg.problem(udp))
    x = [-0.04207567070575896, -0.12626252701191398, -0.5401832679041176, 0.06388017124828997, 
                0.1570632365176983, -0.8471357162115598, -0.11259142034225719, -0.08546452239949272, -0.04200221510495139, 
                -0.6617333706489703, -0.17903139773021548, 0.22614920127948726, 0.2366652945287067, 0.4478005711408385, 
                0.06300561277443284, 0.425970136090571, -0.32632396425541416, 0.23565240320456504, 0.23239777670514036, 
                0.076057597284884, 0.41334839654927047, 0.2314875896061321]
    y = fprob.fun(x)
    print('fval = ' + str(y))
    
def optimize():   
    fprob = single_objective(pg.problem(udp))
    print('interferometer de parallel function evaluation')
       
    # Python Differential Evolution implementation, uses ask/tell for parallel function evaluation.
    ret = de.minimize(fprob.fun, bounds=fprob.bounds, workers=16, popsize=32, max_evaluations=50000)
    
    # Python CMAES implementation, uses ask/tell for parallel function evaluation.
    #ret = cmaes.minimize(fprob.fun, bounds=fprob.bounds, workers=16, popsize=32, max_evaluations=50000)
    
    # Parallel retry using DE    
    #ret = retry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=De_cpp(20000, popsize=31), workers=16)
  
    # Parallel retry using CMA-ES
    #ret = retry.minimize(udp.fitness, bounds=bounds, optimizer=Cma_cpp(20000, popsize=32), workers=16)
 
    # Smart retry using DE
    #ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=De_cpp(1500, popsize=32), workers=16)

    # Smart retry using CMA-ES  
    #ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=Cma_cpp(1500, popsize=32), workers=16)
 
    # Smart retry using DE->CMA sequence  
    #ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=de_cma(1500, popsize=32), workers=16)
    
    print("best result is " + str(ret.fun) + ' x = ' + ", ".join(str(x) for x in ret.x))

if __name__ == '__main__':
    optimize()
    #archipelago()
    # check_good_solution()
    pass