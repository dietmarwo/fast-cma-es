# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and extended by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# requires oagmo (pip install pagmo) for the comparison. Tested with Anaconda 2020.11 
# https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

import pygmo as pg
from interferometryudp import Interferometry
from time import time

def archipelago():    
    print('interferometer sga archipelago')
    uda = pg.sga(gen = 100000)
    #udp = Interferometry(11, './img/orion.jpg', 512)  # scales bad because of CPU cache 
    udp = Interferometry(5, './img/orion.jpg', 32)
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
    
if __name__ == '__main__':
    archipelago()
    pass