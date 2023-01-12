import math

from pykep.core import epoch, fb_prop, SEC2DAY
from kepler.rvt import rvt
      
class resonance:    
    """
    Determines the best "fitting" resonance orbit.
    """

    def __init__(self, planet, rvt_in, rvt_pl,
                 resonances= [[1, 1], [5, 4], [4, 3], [3, 2], [5, 3]]
        ):
        """
        Args:
            - planet (``planet``): resonance planet. 
            - rvt_in: (``rvt``): incoming orbit.
            - rvt_pl: (``rvt``): planet orbit.
            - resonances (``list`` of ``int``): resonance options. 
        """
        assert rvt_in.t == rvt_pl.t # timing must be consistent
        
        self.planet = planet
        self.rvt_in = rvt_in
        self.rvt_pl = rvt_pl       
        self.time = rvt_in.t
        self.resonances = resonances
        self.period = planet.compute_period(epoch(self.time * SEC2DAY))
        self.mu = planet.mu_self
        self.timing_error = -1
        self.rvt_out = None
        self.resonance = None
 
    # useful for debugging
    def __str__(self):
        return str(_resonance) + " " + str(self.timing_error * SEC2DAY) + " " + str(self.rvt_out)
    
    # select a resonance option minimizing the timing error  
    def select_resonance(self, beta, safe_distance):
        v_out = fb_prop(self.rvt_in.v, self.rvt_pl.v, 
                        self.planet.radius + safe_distance, beta, self.mu)
        self.rvt_out = rvt(self.rvt_in.r, v_out, self.time, self.rvt_in.mu)
        period = self.rvt_out.period()
        self.timing_error = math.inf
        for resonance in self.resonances:
            target = self.period * resonance[1] / resonance[0];
            dt = abs(period - target)
            if dt < self.timing_error:
                self.resonance = resonance
                self.timing_error = dt
        return self.timing_error, self.resonance
    
    # time of flight of the resonance transfer
    def tof(self):
        if self.resonance is None:
            raise Exception('_resonance is None')
        return self.resonance[1] * self.period
