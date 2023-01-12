'''
Created on Jan 3, 2023

@author: xxx
'''

import math
from pykep.core import AU, RAD2DEG, SEC2DAY

def normRad(r):
    return r % (2*math.pi)

def get_kepler(a, e, i, W, w, M0, mu, dt = 0):
    M = normRad(M0 + dt * math.sqrt(mu / (a ** 3)))
    ea = _mean2eccentric(M, e)
    kep = Kepler([a, e, i, W, w, ea], mu, dt)
    kep.M0 = M
    # m2 = kep.eccentric2Mean(ea)
    # if m2 < 0:
    #     m2 += 2*math.pi   
    return kep
    
class Kepler:
    
    """
    Keplerian orbit represented by keplerian elements.    
    """
    
    def __init__(self, kep, mu, t):
        """
        Args:
            - kep: keplerian elements  a, e, i, OM, om, E
        """
        self.a = kep[0]
        self.e = kep[1]
        self.i = kep[2]
        self.W = kep[3]
        self.w = kep[4]
        self.ea = kep[5]
        self.mu = mu
        self.t = t

    def copy(self):
        return Kepler(self.kep(), self.mu, self.t)
     
    def __str__(self):
        period = 2 * math.pi * math.sqrt(self.a ** 3 / self.mu)
        apo = self.a * (1 + self.e) / AU
        per = self.a * (1 - self.e) / AU
        return str(apo) + " " + str(per) + " " + \
                str(self.e) + " " + str(self.i * RAD2DEG) + " " + str(period * SEC2DAY)
    
    def kep(self):
        return [self.a, self.e, self.i, self.W, self.w, self.ea]
        
    def propagate_kepler(self, dt):
        M = 0
        if (self.e < 1):
            M = self.ea - self.e * math.sin(self.ea)
            M += math.sqrt(self.mu / (self.a ** 3)) * dt
        else:
            M = self.e * math.tan(self.ea) - math.log(math.tan(0.5 * self.ea() + 0.25 * math.pi))
            M += math.sqrt(abs(self.mu / (self.a ** 3))) * dt
        self.ea = _mean2eccentric(M, self.e)
        self.t += dt
        
    def set_time(self, t):
        self.propagate_kepler(t - self.t)
 
    def minRadius(self):
        return self.a * (1.0 - self.e)

    def maxRadius(self):
        return self.a * (1.0 + self.e)

    def meanMotion(self, mu):
        return math.sqrt(mu / (self.a ** 3))

    def period(self, mu):
        return 2.0 * math.pi / self.meanMotion(mu)

    def periapsis(self):
        return (1.0 - self.e) * self.a

    def apoapsis(self):
        return (1.0 + self.e) * self.a

    def setM0(self, M0, dt, mu):
        ea = self.getEccentricAnomaly(M0, dt, mu)
        self.setEa(ea)
        self.t = dt

    def eccentric2ta(self, E):
        return 2.0 * math.atan(math.tan(0.5 * E) * math.sqrt((1.0 + self.e) / (1.0 - self.e)))

    def getTrueAnomaly(self, M0, dt, mu):
        M = M0 + dt * math.sqrt(mu / (self.a ** 3))
        return self.mean2ta(M)
    
    def getEccentricAnomaly(self, M0, dt, mu):
        M = M0 + dt * math.sqrt(mu / (self.a ** 3))
        return self.mean2eccentric(M)

    def getEccentricAnomaly2(self, M0, dt, mu):
        ta = self.getTrueAnomaly(M0, dt, mu)
        beta = self.e / (1 + math.sqrt((1 - self.e) * (1 + self.e)))
        return ta - 2 * math.atan(beta * math.sin(ta) / (1 + beta * math.cos(ta)))

    def mean2ta(self, M):
        return _mean2ta(M, self.e)

    def mean2eccentric(self, M):
        return _mean2eccentric(M, self.e)
    
    def eccentric2Mean(self, ea):
        return normRad(ea - self.e * math.sin(ea))

from numba import njit

@njit(fastmath=True, cache=True)
def eccentric_anomaly(a, e, M0, mu, dt = 0):
    M = M0 + dt * math.sqrt(mu / (a ** 3))
    return _mean2eccentric(M, e)

@njit(fastmath=True, cache=True)
def _mean2ta(M, e):
    Mnorm = M % (2 * math.pi)
    E = keplerStart3(e, Mnorm)
    for _ in range(50):
        delta = eps3(e, Mnorm, E)
        E -= delta
        if abs(delta) < 1.0e-12: break
    return 2.0 * math.atan(math.tan(0.5 * E) * math.sqrt((1.0 + e) / (1.0 - e)))

@njit(fastmath=True, cache=True)
def _mean2eccentric(M, e):
    Mnorm = M % (2 * math.pi)
    E = keplerStart3(e, Mnorm)
    for _ in range(50):
        delta = eps3(e, Mnorm, E)
        E -= delta
        if abs(delta) < 1.0e-12: break
    return ((E + math.pi) % (2 * math.pi)) - math.pi

@njit(fastmath=True, cache=True)
def keplerStart3(e, M):
    t34 = e * e
    t35 = e * t34
    t33 = math.cos(M)
    return M + (-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * math.sin(M)

@njit(fastmath=True, cache=True)
def eps3(e, M, x):
    t1 = math.cos(x)
    t2 = -1.0 + e * t1
    t3 = math.sin(x)
    t4 = e * t3
    t5 = -x + t4 + M
    t6 = t5 / ((0.5 * t5) * t4 / t2 + t2)
    return t5 / ((0.5 * t3 - t1 * t6 / 6.0) * e * t6 + t2)
    
if __name__ == '__main__':
    pass