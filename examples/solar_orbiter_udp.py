# This code is derived from https://github.com/esa/pykep/pull/127 
# originally developed by Moritz v. Looz @mlooz . 
# It was modified following suggestions from Waldemar Martens @MartensWaldemar_gitlab
# In this implementation there are restrictions regarding the allowed planet 
# sequence which will be removed in later revisions.
# The code is designed around an "orbit abstraction" class _rvt simplifying the
# definition of the objective function. 
#
# This Python code is about factor 3.3 slower than the equivalent Java code.
# https://github.com/dietmarwo/fcmaes-java/tree/master/src/main/java/fcmaes/examples/Solo.java 
# 
# This problem is quite a challenge for state of the art optimizers, but
# good solutions fulfilling the requirements can be found.
# See https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter

from math import cos, pi, sin, sqrt
import math
from typing import List

from pykep import AU, DAY2SEC, DEG2RAD, RAD2DEG, SEC2DAY, epoch, ic2par
from pykep.core import fb_prop, fb_vel, lambert_problem, propagate_lagrangian
from pykep.planet import jpl_lp

import ctypes as ct
import multiprocessing as mp
import numpy as np

bval = mp.RawValue(ct.c_double, 1E99)

safe_distance = 350000

# select lowest dv lambert considering GA at planet
class lambert_problem_multirev_ga:
 
    def __init__(self, v_in, lp, planet, v_planet):
        best_i = 0        
        n = len(lp.get_v1())
        if n > 0: 
            best_dv = math.inf           
            for i in range(n):
                vin = [a - b for a, b in zip(v_in, v_planet)]
                vout = [a - b for a, b in zip(lp.get_v1()[i], v_planet)]
                dv = fb_vel(vin, vout, planet)
                if dv < best_dv:
                    best_dv = dv
                    best_i = i 
        self.best_i = best_i
        self.lambert_problem = lp
        
    def get_v1(self):
        return [self.lambert_problem.get_v1()[self.best_i]]
   
    def get_v2(self):
        return [self.lambert_problem.get_v2()[self.best_i]]

    def get_r1(self):
        return self.lambert_problem.get_r1()  

    def get_r2(self):
        return self.lambert_problem.get_r2()  

    def get_mu(self):
        return self.lambert_problem.get_mu()

    def get_x(self):
        return [self.lambert_problem.get_x()[self.best_i]]

    def get_iters(self):
        return [self.lambert_problem.get_iters()[self.best_i]]

    def get_tof(self):
        return self.lambert_problem.get_tof()

    def get_Nmax(self):
        return self.lambert_problem.get_Nmax()

# keplerian orbit represented by r, v, time and mu    
class _rvt:
    
    def __init__(self, r, v, time, mu):
        self._r = r
        self._v = v
        self._t = time
        self._mu = mu
        
    def __str__(self):
        a, e, i, _, _, _ = self.kepler()
        period = 2 * pi * sqrt(a ** 3 / self._mu)
        apo = a * (1 + e) / AU
        per = a * (1 - e) / AU
        return str(self._r) + " " + str(self._v) + " " + str(self._t) + " " +  \
                str(apo) + " " + str(per) + " " + \
                str(e) + " " + str(i*RAD2DEG) + " " + str(period*SEC2DAY)
        
    def propagate(self, dt):
        rvt = _rvt(self._r, self._v, self._t, self._mu)
        rvt._r, rvt._v = propagate_lagrangian(rvt._r, rvt._v, DAY2SEC*dt, self._mu)
        return rvt
            
    def kepler(self):
        return ic2par(self._r, self._v, self._mu) 
    
    
    def plot(self, tof, N=60, units=AU, color="b", label=None, axes=None):
        from pykep.orbit_plots import plot_kepler
        plot_kepler(r0 = self._r, v0 = self._v, tof = DAY2SEC*tof, 
                    mu = self._mu, N=N, units=units, color=color, 
                    label=label, axes=axes)

    def period(self):
        kep = ic2par(self._r, self._v, self._mu) 
        a = kep[0]
        meanMotion = sqrt(self._mu / (a*a*a))
        return 2.0 * math.pi / meanMotion;
    
    def rotate(self, k, theta):
        rvt = _rvt(self._r, self._v, self._t, self._mu)
        rvt._r = rotate_vector(self._r, k, theta)
        rvt._v = rotate_vector(self._v, k, theta)
        return rvt 

# determines the best "fitting" resonance orbit      
class _resonance:

    def __init__(self, pl, rvt_in, rvt_pl, resonances):
        self._pl = pl
        self._rvt_in = rvt_in
        self._rvt_pl = rvt_pl       
        self._time = rvt_in._t
        self._resonances = resonances
        self._period = pl.compute_period(epoch(self._time))
        self._mu = pl.mu_self
        self._dt = -1
        self._rvt_out = None
        self._resonance = None
 
    def __str__(self):
        return str(_resonance) + " " + str(self._dt) + " " + str(self._rvt_out)
       
    def select(self, beta):
        v_out = fb_prop(self._rvt_in._v, self._rvt_pl._v, 
                        self._pl.radius + safe_distance, beta, self._mu)
        self._rvt_out = _rvt(self._rvt_in._r, v_out, self._time, self._rvt_in._mu)
        period = self._rvt_out.period()
        self._dt = math.inf
        for resonance in self._resonances:
            target = self._period * resonance[1] / resonance[0];
            dt = abs(period - target)
            if dt < self._dt:
                self._resonance = resonance
                self._dt = dt
        return self._dt, self._resonance
    
    def tof(self):
        return self._resonance[1] * self._period * SEC2DAY

# propagate rvt_outs, rvt_ins, rvt_pls, dvs using resonance        
def _compute_resonance(pl, resonances, beta, used_resos, reso_dts, rvt_outs, 
                       rvt_ins, rvt_pls, dvs, resos = None):
    rvt_in = rvt_ins[-1] # current spaceship
    rvt_pl = rvt_pls[-1] # current planet
    reso = _resonance(pl, rvt_in, rvt_pl, resonances)
    reso_dt, used_reso = reso.select(beta)
    if not resos is None:
        resos.append(reso)
    used_resos.append(used_reso)
    reso_dts.append(reso_dt)
    rvt_outs.append(reso._rvt_out)
    tof = reso.tof()
    time2 = reso._time + tof
    rvt_pl2 = _rvt_planet(pl, time2)
    rvt_pls.append(rvt_pl2) 
    rvt_in2 = _rvt(rvt_pl2._r, reso._rvt_out._v, time2, rvt_pl2._mu) # its a resonance, we arrive with same v as we started
    rvt_ins.append(rvt_in2)
    dvs.append(0) # # its a resonance, we don't need an impulse
            
def _rvt_planet(pl, time):
    r, v = pl.eph(epoch(time))
    return _rvt(r, v, time, pl.mu_central_body)

# propagate rvt_outs, rvt_ins, rvt_pls, dvs using MGA / Lambert
def _dv_mga(pl1, pl2, tof, max_revs, rvt_outs, rvt_ins, rvt_pls, dvs, lps = None):
    rvt_pl = rvt_pls[-1] # current planet
    v_in =  rvt_pl._v if rvt_ins[-1] is None else rvt_ins[-1]._v
    rvt_pl2 = _rvt_planet(pl2, rvt_pl._t + tof)        
    rvt_pls.append(rvt_pl2)
    r = rvt_pl._r
    vpl = rvt_pl._v
    r2 = rvt_pl2._r
    lp = lambert_problem(r, r2, tof * DAY2SEC, rvt_pl._mu, False, max_revs)
    lp = lambert_problem_multirev_ga(v_in, lp, pl1, vpl)
    if not lps is None:
        lps.append(lp)
    v_out = lp.get_v1()[0]
    rvt_out = _rvt(r, v_out, rvt_pl._t, rvt_pl._mu)
    rvt_outs.append(rvt_out)
    rvt_in = _rvt(r2, lp.get_v2()[0], rvt_pl._t + tof, rvt_pl._mu)
    rvt_ins.append(rvt_in)
    vr_in = [a - b for a, b in zip(v_in, vpl)]
    vr_out = [a - b for a, b in zip(v_out, vpl)]
    dv = fb_vel(vr_in, vr_out, pl1)
    dvs.append(dv)

def rotate_vector(v, k, theta):
    dP = np.dot(k, v)
    cosTheta = cos(theta)
    sinTheta = sin(theta)
    # rotate vector into coordinate system defined by the sun's equatorial plane
    # using Rodrigues rotation formula
    r_rot = [
        a * cosTheta + b * sinTheta + c * (1 - cosTheta) * dP
        for a, b, c in zip(v, np.cross(k, v), k)
    ]
    return r_rot

class _solar_orbiter_udp:

    earth = jpl_lp("earth")
    venus = jpl_lp("venus")

    def __init__(
        self,
        t0=[epoch(0), epoch(10000)],
        max_revs: int = 3,
        resos = [[[1,1], [5,4], [4,3]],
                [[1,1], [5,4], [4,3]],
                [[5,4], [4,3], [3,2], [5,3]],
                [[4,3], [3,2], [5,3]],
                [[4,3], [3,2], [5,3]],
                [[3,2], [5,3]]],
        seq = [earth, venus, venus, earth, venus, venus, venus, venus, venus, venus],
    ):
        """
        Args:
            - max_revs (``int``): maximal number of revolutions for lambert transfer for VE, EV transfers
            - resos: list of allowed resonances for the VV transfers
            - seq (``list``)
        """

        tof = [[50, 400]] * 3 # only EV and VE transfers
        self._max_mission_time = 11.0*365.25
        self._max_dv0 = 5600
        self._min_beta = -math.pi
        self._max_beta = math.pi
        
        self._seq = seq
        self._resos = resos
        self._t0 = t0
        self._tof = tof
        self._max_revs = max_revs       
        self._n_legs = len(seq) - 1
 
        # initialize data to compute heliolatitude
        t_plane_crossing = epoch(7645)
        rotation_axis = seq[0].eph(t_plane_crossing)[0]
        self._rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        self._theta = 7.25 * DEG2RAD
         
    def _compute_dvs(self, x: List[float], lps = None, resos = None):
        
        t0 = x[0]
        tof01 = x[1]
        tof23 = x[2]
        tof34 = x[3]
        betas = x[4:]
       
        rvt_outs = []
        rvt_ins = [None] # no rvt_in at first planet
        rvt_pls = []
        dvs = []
        used_resos = []
        reso_dts = []
        
        rvt_pls.append(_rvt_planet(self._seq[0], t0))
 
        _dv_mga(self._seq[0], self._seq[1], tof01, self._max_revs, rvt_outs, rvt_ins, rvt_pls, dvs, lps)
         
        _compute_resonance(self._seq[1], self._resos[0], betas[0], used_resos, reso_dts, 
                                 rvt_outs, rvt_ins, rvt_pls, dvs, resos)
        
        _dv_mga(self._seq[2], self._seq[3], tof23, self._max_revs, rvt_outs, rvt_ins, rvt_pls, dvs, lps)
  
        _dv_mga(self._seq[3], self._seq[4], tof34, self._max_revs, rvt_outs, rvt_ins, rvt_pls, dvs, lps)

        for i in range(1,6):            
            _compute_resonance(self._seq[i+3], self._resos[i], betas[i], used_resos, reso_dts, 
                                     rvt_outs, rvt_ins, rvt_pls, dvs, resos)
        
        rvt_in = rvt_ins[-1] # we arrive
        rvt_pl = rvt_pls[-1] # at planet
        # add GA using betas[6] and safe radius
        v_out = fb_prop(rvt_in._v, rvt_pl._v, self._seq[8].radius + safe_distance, 
                        betas[6], self._seq[8].mu_self) 
        rvt_out = _rvt(rvt_in._r, v_out, rvt_in._t, rvt_in._mu)
        # rotate to get inclination respective to solar equator. 
        rvt_outs.append(rvt_out.rotate(self._rotation_axis, self._theta))

        n = len(rvt_ins)
        assert len(rvt_outs) == n  # we added a final rvt_out
        assert len(rvt_pls) == n 
        assert len(dvs) == n - 1
        return rvt_outs, rvt_ins, rvt_pls, reso_dts, dvs

    # Objective function
    def fitness(self, x):
        if len(x) != len(self.get_bounds()[0]):
            raise ValueError(
                "Expected "
                + str(len(self.get_bounds()[0]))
                + " parameters but got "
                + str(len(x))
            )

        rvt_outs, rvt_ins, rvt_pls, reso_dts, dvs = self._compute_dvs(x)
        # compute final flyby and resulting trajectory
        rvt_out = rvt_outs[-1] # we arrive, its already rotated
        a, e, incl, W, w, E = rvt_out.kepler()
        final_perihelion = a * (1 - e)
        # orbit should be as polar as possible, but we do not care about prograde/retrograde
        corrected_inclination = abs(abs(incl) % pi - pi / 2) * RAD2DEG

        # check perihelion and aphelion bounds during the flight
        min_sun_distance = final_perihelion
        max_sun_distance = AU

        for i in range(len(rvt_outs)-1):
            rvt0 = rvt_outs[i]
            rvt1 = rvt_outs[i+1]
            t0 = rvt0._t
            t1 = rvt1._t
            dt = (t1 - t0) * DAY2SEC
            transfer_a, transfer_e, _, _, _, _ = rvt0.kepler()
            transfer_period = 2 * pi * sqrt(transfer_a ** 3 / rvt0._mu)
            
            # print (dt*SEC2DAY, transfer_period*SEC2DAY, 
                   # transfer_a * (1 + transfer_e)/AU, transfer_a * (1 - transfer_e)/AU, rvt0)

            # update min and max sun distance
            if dt > transfer_period:
                max_sun_distance = max(max_sun_distance, transfer_a * (1 + transfer_e))
                min_sun_distance = min(min_sun_distance, transfer_a * (1 - transfer_e))
                    
        # overall time limit
        time_all = (rvt_outs[-1]._t - rvt_outs[0]._t)
        time_val = time_all
        time_limit = self._max_mission_time # 11 years
        if time_val > time_limit:
            time_val += 10 * (time_val - time_limit) 
        
        perihelion_val = final_perihelion / AU
        # avoid bonus for perihelion < 0.28
        if perihelion_val < 0.28:
            perihelion_val += 10*(0.28 - perihelion_val);
        
        # wrong reso timing in seconds
        reso_penalty = np.sum(reso_dts)   
        
        # wrong minimal / maximal distance
        distance_penalty = max(0, 0.28 - min_sun_distance / AU)  
        distance_penalty += max(0, max_sun_distance / AU - 1.2)  
              
        # allow start dv    
        dvs[0] = max(0, dvs[0] - self._max_dv0) 
        dv_val = np.sum(dvs)
               
        value = (100 * dv_val + 
                100 * corrected_inclination + 
                5000 * (perihelion_val - 0.28) +
                0.5 * time_val + 
                reso_penalty +
                50000 * distance_penalty
                )

        if value < bval.value:
            bval.value = value
            print(str(value) 
                  + " " + str(incl * RAD2DEG) 
                  + " " + str(time_all / 365.25)  
                  + " " + str(dv_val) 
                  + " " + str(min_sun_distance / AU) 
                  + " " + str(max_sun_distance / AU) 
                  + " " + str(reso_penalty) + " "
                  + " " + str(distance_penalty)
                  )
   
        return [value]
 
    def get_nobj(self):
        return 1

    def get_bounds(self):
        t0 = self._t0
        tof = self._tof
        n_legs = self._n_legs
        
        lb = [t0[0]]
        ub = [t0[1]]
        lb += [t[0] for t in tof]
        ub += [t[1] for t in tof]
        
        nbetas = n_legs - len(tof) + 1
        lb += [self._min_beta] * nbetas
        ub += [self._max_beta] * nbetas
        return (lb, ub)

    def get_nic(self):
        return 0

    def pretty(self, x):
        lambert_legs = []
        resos = []
        rvt_outs, rvt_ins, rvt_pls, reso_dts, dvs = self._compute_dvs(x, lambert_legs, resos)              
        ep = [epoch(rvt_pl._t) for rvt_pl in rvt_pls]
        b_legs = [[rvt_out._r, rvt_out._v] for rvt_out in rvt_outs]
        Vinfx, Vinfy, Vinfz = [
            a - b for a, b in zip(b_legs[0][1], self._seq[0].eph(ep[0])[1])
        ]
        common_mu = rvt_outs[0]._mu
      

        lambert_indices = [lam.best_i for lam in lambert_legs]

        print("Multiple Gravity Assist (MGA) + Resonance problem: ")
        print("Planet sequence: ", [pl.name for pl in self._seq])

        print("Departure: ", self._seq[0].name)
        print("\tEpoch: ", ep[0], " [mjd2000]")
        print("\tSpacecraft velocity: ", b_legs[0][1], "[m/s]")
        print("\tLaunch velocity: ", [Vinfx, Vinfy, Vinfz], "[m/s]")
        _, _, transfer_i, _, _, _ = ic2par(*(b_legs[0]), common_mu)
        print("\tOutgoing Inclination:", transfer_i * RAD2DEG, "[deg]")
        print("\tNumber of Revolutions:", int((lambert_indices[0] + 1) / 2))
        print("\tLambert Index:", int(lambert_indices[0]))
        
        lambert_i = 0
        reso_i = 0
        #assert len(DV) == len(self._seq) - 2
        for i in range(1, len(self._seq)):
            pl = self._seq[i]
            e = ep[i]
            dv = dvs[i] if i < len(self._seq)-1 else 0
            leg = b_legs[i]
            print("Fly-by: ", pl.name)
            print("\tEpoch: ", e, " [mjd2000]")
            print("\tDV: ", dv, "[m/s]")
            eph = pl.eph(e)
            if i < len(self._seq)-1: # last one is roteted
                assert np.linalg.norm([a - b for a, b in zip(leg[0], eph[0])]) < 0.01
            _, _, transfer_i, _, _, _ = ic2par(eph[0], leg[1], common_mu)
            print("\tOutgoing Inclination:", transfer_i * RAD2DEG, "[deg]")
            if pl != self._seq[i-1]: # no lamberts for resonances
                print("\tLambert Index:", str(lambert_indices[lambert_i]))
                lambert_i += 1
            else: # resonance at Venus
                print("\tResonance:", str(resos[reso_i]._resonance))
                print("\tResonance time error:", str(resos[reso_i]._dt) + " sec")
                reso_i += 1
               
        print("Final Fly-by: ", self._seq[-1].name)
        print("\tEpoch: ", ep[-1], " [mjd2000]")
        print("\tSpacecraft velocity: ", rvt_outs[-1]._v, "[m/s]")
        print("\tBeta: ", x[-1])
        print("\tr_p: ", self._seq[-1].radius + safe_distance)

        print("Resulting Solar orbit:")
        a, e, i, W, w, E = rvt_outs[-1].kepler()
        print("Perihelion: ", (a * (1 - e)) / AU, " AU")
        print("Aphelion: ", (a * (1 + e)) / AU, " AU")
        print("Inclination: ", i * RAD2DEG, " degrees")
        T = [rvt_outs[i+1]._t - rvt_outs[i]._t for i in range(len(rvt_outs)-1)]
        print("Time of flights: ", T, "[days]")

    def plot(self, x, axes=None, units=AU, N=60):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet

        rvt_outs, _, rvt_pls, _, _ = self._compute_dvs(x)              
        ep = [epoch(rvt_pl._t) for rvt_pl in rvt_pls]
 
        # Creating the axes if necessary
        if axes is None:
            mpl.rcParams["legend.fontsize"] = 10
            fig = plt.figure()
            axes = fig.gca(projection="3d")

        plt.xlim([-1, 1])
        # plt.axis('equal')
        # planets
        for pl, e in zip(self._seq, ep):
            plot_planet(
                pl, e, units=units, legend=True, color=(0.7, 0.7, 1), axes=axes
            )
            
        # lamberts and resonances    
        for i in range(0, len(self._seq)):
            pl = self._seq[i]
            # stay at planet: it is a resonance colored black
            is_reso = i == len(self._seq) - 1 or pl == self._seq[i+1]
            rvt_out = rvt_outs[i]
            tof = 300 if i == len(self._seq) - 1 else rvt_outs[i+1]._t - rvt_out._t
            rvt_out.plot(tof, 
                units=units,
                N=4*N,
                color= "k" if is_reso else "r",
                axes=axes)
        return axes

    def eph(self, rvts, t):
        for i in range(0, len(rvts)):
            rvt = rvts[i]
            if i == len(self._seq) - 1 or rvts[i+1]._t > t:
                dt = t - rvt._t
                rvt = rvt.propagate(dt)
                return rvt._r, rvt._v
                
    def plot_distance_and_flybys(self, x, axes=None, N=1200, extension=300):
        import matplotlib.pyplot as plt
        rvt_outs, _, rvt_pls, _, _ = self._compute_dvs(x)              
        ep = [rvt_pl._t for rvt_pl in rvt_pls]
        T = [rvt_outs[i+1]._t - rvt_outs[i]._t for i in range(len(rvt_outs)-1)]
        timeframe = np.linspace(0, sum(T) + extension, N)            
        earth = self._seq[0]
        venus = self._seq[-1]

        distances = []
        edistances = []
        vdistances = []

        for day in timeframe:
            t = x[0] + day
            pos, _ = self.eph(rvt_outs, t)
            epos, _ = earth.eph(t)
            vpos, _ = venus.eph(t)
            distances.append(np.linalg.norm(pos) / AU)
            edistances.append(np.linalg.norm(epos) / AU)
            vdistances.append(np.linalg.norm(vpos) / AU)

        fl_times = list()
        fl_distances = list()
        for pl, t in zip(self._seq, ep):
            fl_times.append(t - x[0])
            pos, _ = pl.eph(t)
            fl_distances.append(np.linalg.norm(pos) / AU)

        if axes is None:
            fig, axes = plt.subplots()
        plt.ylim([0, 1.3])

        axes.plot(list(timeframe), distances, label="Solar Orbiter")
        axes.plot(list(timeframe), edistances, label="Earth")
        axes.plot(list(timeframe), vdistances, label="Venus")
        plt.scatter(fl_times, fl_distances, marker="o", color="r")
        axes.set_xlabel("Days")
        axes.set_ylabel("AU")
        axes.set_title("Distance to Sun")
        axes.legend()
        return axes

