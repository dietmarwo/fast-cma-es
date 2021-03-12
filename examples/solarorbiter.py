# This code is derived from https://github.com/esa/pykep/pull/127 
# originally developed by Moritz v. Looz @mlooz . 
# It was modified following suggestions from Waldemar Martens @MartensWaldemar_gitlab

# Solar orbiter is quite a challenge for state of the art optimizers, but
# good solutions fulfilling the requirements can be found and an example is
# shown in check_good_solution()
#
# See https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter

import time
from pykep.planet import jpl_lp
from solar_orbiter_udp import _solar_orbiter_udp
import matplotlib.pyplot as plt

import pygmo as pg

from fcmaes import retry, advretry 
from fcmaes.optimizer import logger, de_cma, single_objective, de, Bite_cpp
    
# Other imports
tmin = time.time() / (24*3600) - 30*365 -7 + 2/24 - 2*365
tmax = time.time() / (24*3600) - 30*365 -7 + 2/24 + 2*365

logger("solarorbiter.log")

def check_good_solution():
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    seq = [earth, venus, venus, earth, venus, venus, venus, venus, venus, venus]
    solar_orbiter = _solar_orbiter_udp([tmin, tmax], seq=seq)  
    prob = pg.problem(solar_orbiter)
    x = [7454.820505282011, 399.5883816298621, 161.3293044402143, 336.35353340379817, 0.16706526043179085, -2.926263900573538, 
         2.1707384653871475, 3.068749728236526, 2.6458336313296913, 3.0472278514692377, 2.426804445518446]  
    print (prob.fitness(x)) 
    solar_orbiter.pretty(x)
    solar_orbiter.plot(x)
    solar_orbiter.plot_distance_and_flybys(x)
     
def optimize():   
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    seq = [earth, venus, venus, earth, venus, venus, venus, venus, venus, venus]
    solar_orbiter = _solar_orbiter_udp([tmin, tmax], seq=seq)  
    prob = pg.problem(solar_orbiter)
    fprob = single_objective(prob)   
       
    # logger().info('solar orbiter' + ' de -> cmaes c++ smart retry')
    # ret = advretry.minimize(fprob.fun, bounds=fprob.bounds, num_retries = 3000, 
        # logger = logger(), optimizer=de_cma(1500))
    
    logger().info('solar orbiter' + ' BiteOpt parallel retry')
    ret = retry.minimize(fprob.fun, bounds=fprob.bounds, num_retries = 32000, 
                         logger = logger(), optimizer=Bite_cpp(100000, M=6))
    return ret

if __name__ == '__main__':
    # optimize()
    check_good_solution()   
    plt.show()

    pass