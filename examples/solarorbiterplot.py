# requires pykep PR branch for 
# https://github.com/esa/pykep/pull/127

# plots results of good sequences determined by solarorbitermulti.py,

from math import acos
import time

from numpy import sign
from numpy.linalg import norm
from pykep import AU, epoch
from pykep.planet import jpl_lp
from pykep.trajopt.gym._solar_orbiter import _solar_orbiter_udp

import matplotlib.pyplot as plt
import pygmo as pg

# Other imports
tmin = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 - 2*365)
tmax = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 + 2*365)

def names(seq):
    return " ".join((p.name) for p in seq)

def plot_solar_orbiter():
    
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")

    #fval = 1.72
    seq = [earth, earth, venus, earth, venus, venus, earth, venus, venus]
    pop_champion_x = [7002.958769462786, 730.5123955221125, 272.41965071272784, 74.1806705492443, 171.97750009574423, 375.5928357818143, 682.4987974937061, 93.3897864500331, 341.2809749435019, 0.9207077095318533, 1.0578321216244067]    

    #fval = 1.86
#     seq = [earth, earth, venus, venus, earth, venus, venus, venus, venus]
#     pop_champion_x = [7938.056736711781, 666.628077582916, 106.09878929364706, 449.4002461984736, 62.505131189084054, 404.2791217115739, 674.0850051741737, 674.0850908654102, 410.89560639110823, 2.175328584308378, 1.05783768709899]

    #fval = 1.87
#     seq = [earth, venus, venus, earth, venus, venus, venus, venus, venus]
#     pop_champion_x = [7309.955686504108, 242.00305306309562, 403.31913927172866, 80.568048104212, 376.3052464119222, 672.2642026142728, 672.2698205678452, 672.2914624962455, 410.3229994185495, 0.9756176647367582, 1.0578321217798627]
    
    #fval = 2.05
#     seq = [earth, venus, venus, venus, venus, earth, venus, venus, venus]
#     pop_champion_x = [7876.843342769882, 232.52423111093282, 432.08683385632736, 432.10172802885165, 432.1101414935594, 387.8659887573876, 81.40752718157002, 671.8630715210193, 410.8561237980522, 2.7335637831912654, 19.244826411431877]

    solar_orbiter = _solar_orbiter_udp([tmin, tmax], seq=seq)
    prob = pg.problem(pg.unconstrain(solar_orbiter,method="weighted",weights=[1.0, 10.0, 100, 100]))
    fval = prob.fitness(pop_champion_x) 
    print('sequence ' + names(seq))
    print('fval = ' + str(fval))
        
    solar_orbiter.pretty(pop_champion_x)
    solar_orbiter.plot(pop_champion_x)
    
    # Plot solar distance in AE
    timeframe = range(1,5*365)
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    
    distances = []
    edistances = []
    vdistances = []
    
    for i in timeframe:
        epoch = pop_champion_x[0]+i
        pos, vel = solar_orbiter.eph(pop_champion_x, epoch)
        epos, evel = earth.eph(epoch)
        vpos, vvel = venus.eph(epoch)
        distances.append(norm(pos) / AU)
        edistances.append(norm(epos) / AU)
        vdistances.append(norm(vpos) / AU)
    
    fig, ax = plt.subplots()
    ax.plot(list(timeframe), distances, label="Solar Orbiter")
    ax.plot(list(timeframe), edistances, label="Earth")
    ax.plot(list(timeframe), vdistances, label="Venus")
    ax.set_xlabel("Days")
    ax.set_ylabel("AU")
    ax.set_title("Distance to Sun")
    ax.legend()
    
    # Plot inclination and distance
    inclinations = []
    for i in timeframe:
        epoch = pop_champion_x[0]+i
        pos, _ = solar_orbiter.eph(pop_champion_x, epoch)
        inclination = sign(pos[2])*acos(norm(pos[:2]) / norm(pos))
        inclinations.append(inclination)
    
    color = 'tab:red'
    fig2, ax2 = plt.subplots()
    ax2.plot(list(timeframe), inclinations, color=color)
    ax2.set_ylabel("Inclination", color=color)
    ax2.set_xlabel("Days")
    ax.set_title("Distance and Inclination")
    
    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax3.set_ylabel('AU', color=color)
    ax3.plot(list(timeframe), distances, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
if __name__ == '__main__':
    plot_solar_orbiter()
    pass