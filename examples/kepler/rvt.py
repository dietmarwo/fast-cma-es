from math import pi, sqrt, cos, sin
import math

from pykep.core import AU, RAD2DEG, SEC2DAY
from pykep.core.core import propagate_lagrangian, ic2par, epoch, \
    propagate_taylor

import numpy as np
from kepler.kepler import Kepler
from pykep.core.core import ic2par, par2ic
 
class rvt:
    
    """
    Keplerian orbit represented by radius, velocity, time and mu.    
    """
    
    def __init__(self, r, v, time, mu):
        """
        Args:
            - r (``tuple`` of ``float``): cartesian position in m.
            - v: (``tuple`` of ``float``): velocity in m/s.
            - time: (``float``): time in seconds.
            - mu (`float``): gravity parameter of the central body.
        """

        self.r = r
        self.v = v
        self.t = time  # in seconds
        self.mu = mu

    # useful for debugging        
    def __str__(self):
        a, e, i, _, _, _ = self.kepler()
        period = 2 * pi * sqrt(a ** 3 / self.mu)
        apo = a * (1 + e) / AU
        per = a * (1 - e) / AU
        return str(self.r) + " " + str(self.v) + " " + str(self.t * SEC2DAY) + " " + \
                str(apo) + " " + str(per) + " " + \
                str(e) + " " + str(i * RAD2DEG) + " " + str(period * SEC2DAY)
    
    def apply_dv(self, dv):
        v = [a + b for a, b in zip(self.v, dv)]
        return rvt(self.r, v, self.t, self.mu)
        
    def propagate_lagrangian(self, tof):
        orb = rvt(self.r, self.v, self.t + tof, self.mu)
        orb.r, orb.v = propagate_lagrangian(orb.r, orb.v, tof, self.mu)
        return orb
 
    def propagatetaylor(self, tof, m0, thrust, veff=1, log10tol=-15, log10rtol=-15):
        orb = rvt(self.r, self.v, self.t + tof, self.mu)
        orb.r, orb.v, m = propagate_taylor(orb.r, orb.v, m0, thrust, tof, self.mu,
                                             veff, log10tol, log10rtol)
        return orb, m
    
    # keplarian parameters a, e, i, W, w, E       
    def kepler(self):
        return ic2par(self.r, self.v, self.mu) 
    
    def get_kepler(self):
        kep = self.kepler()
        return Kepler(kep, self.mu, self.t)
        
    # plots orbit from current time up to time + tof   
    def plot(self, tof, N=60, units=AU, color="b", label=None, axes=None):
        from pykep.orbit_plots import plot_kepler
        plot_kepler(r0=self.r, v0=self.v, tof=tof,
                    mu=self.mu, N=N, units=units, color=color,
                    label=label, axes=axes)

    def period(self):
        kep = ic2par(self.r, self.v, self.mu) 
        a = kep[0]
        meanMotion = sqrt(self.mu / (a ** 3))
        return 2.0 * math.pi / meanMotion;  # in seconds
    
    def rotate(self, k, theta):
        orb = rvt(self.r, self.v, self.t, self.mu)
        orb.r = rotate_vector(self.r, k, theta)
        orb.v = rotate_vector(self.v, k, theta)
        return orb 
    
    def tof(self, rvt2):
        return rvt2.t - self.t
    
    def copy(self):
        return rvt(self.r, self.v, self.t, self.mu)

def rvt_kepler(kep):    
    rv = par2ic(kep.kep(), kep.mu)
    return rvt(rv[0], rv[1], kep.t, kep.mu)

def rvt_planet(pl, time):
    r, v = pl.eph(epoch(time * SEC2DAY))
    return rvt(r, v, time, pl.mu_central_body)

def rotate_vector(v, k, theta):
    dP = np.dot(k, v)
    cosTheta = cos(theta)
    sinTheta = sin(theta)
    # rotate using Rodrigues rotation formula
    rrot = [
        a * cosTheta + b * sinTheta + c * (1 - cosTheta) * dP
        for a, b, c in zip(v, np.cross(k, v), k)
    ]
    return rrot

