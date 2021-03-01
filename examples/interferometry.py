# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and is extended here by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# Tested with Anaconda 2020.11 https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

import pygmo as pg
from interferometryudp import Interferometry
from fcmaes import de, cmaes, retry, advretry
from fcmaes.optimizer import single_objective, de_cma_py, Cma_python, De_python, Cma_cpp
    
def optimize():   
    #udp = Interferometry(11, './img/orion.jpg', 512) # scales bad because of CPU cache 
    udp = Interferometry(5, './img/orion.jpg', 32)
    fprob = single_objective(pg.problem(udp))
    print('interferometer de parallel function evaluation')
    
    # Python Differential Evolution implementation, uses ask/tell for parallel function evaluation.
    ret = de.minimize(fprob.fun, bounds=fprob.bounds, workers=8, popsize=31, max_evaluations=50000)
    
    # Python CMAES implementation, uses ask/tell for parallel function evaluation.
    #ret = cmaes.minimize(fprob.fun, bounds=fprob.bounds, workers=8, popsize=31, max_evaluations=20000)
    
    # Parallel retry using DE    
    #ret = retry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=De_python(20000, popsize=31), workers=8)
  
    # Parallel retry using CMA-ES
    #ret = retry.minimize(udp.fitness, bounds=bounds, optimizer=Cma_cpp(20000, popsize=31), workers=8)
 
    # Smart retry using DE
    #ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=De_python(1500, popsize=31), workers=8)

    # Smart retry using CMA-ES  
    #ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=Cma_cpp(1500, popsize=31), workers=8)
     
    print("best result is " + str(ret.fun) + ' x = ' + ", ".join(str(x) for x in ret.x))

if __name__ == '__main__':
    optimize()
    pass