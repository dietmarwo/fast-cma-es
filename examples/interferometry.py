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
from fcmaes.optimizer import single_objective, de_cma_py, Cma_python, De_python, Cma_cpp, De_cpp, de_cma, Bite_cpp

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
    # fval = 98.227 can you find a better solution?
    x = [ 0.7926460031808127, -0.01650655233593057, -0.14134206299527174,
            -0.16791222140510897, -0.002633945207571764, -0.06405122912190953, 0.4165681972725821,
            -0.2078626908712642, -0.3339022482767109, 0.5939070738122283, 0.32569734809318374, -0.3097754725415154,
            -0.08627056373307851, -0.08775017156974396, -0.08037736247731875, -0.08404567377996597,
            -0.10855640502975383, -0.0657328539015265, -0.0705920259815862, -0.2563975939710659,
            -0.09779551032823308, -0.08509767183006334 ]        
    y = fprob.fun(x)
    print('fval = ' + str(y))
    
def optimize():   
    fprob = single_objective(pg.problem(udp))
    print('interferometer optimization')
       
    # Python Differential Evolution implementation, uses ask/tell for parallel function evaluation.
    ret = de.minimize(fprob.fun, bounds=fprob.bounds, workers=16, popsize=32, max_evaluations=50000)
    
    # Python CMAES implementation, uses ask/tell for parallel function evaluation.
    #ret = cmaes.minimize(fprob.fun, bounds=fprob.bounds, workers=16, popsize=32, max_evaluations=50000)
    
    # Parallel retry using DE    
    #ret = retry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=De_cpp(20000, popsize=32), workers=16, num_retries=64)

    # Parallel retry using Bite    
    # ret = retry.minimize(fprob.fun, bounds=fprob.bounds, optimizer=Bite_cpp(20000, M=1), workers=16, num_retries=64)

    # Parallel retry using CMA-ES
    #ret = retry.minimize(udp.fitness, bounds=bounds, optimizer=Cma_cpp(20000, popsize=32), workers=16, num_retries=64)
 
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
    #check_good_solution()
    pass