# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import math
import os
import numpy as np
import ctypes as ct
from scipy.optimize import Bounds

basepath = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith('linux'):
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libgtoplib.so')    
elif 'mac' in sys.platform:
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libgtoplib.dylib')  
else:
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libgtoplib.dll')

class Astrofun(object):
    """Provides access to ESAs GTOP optimization test functions."""
    def __init__(self, name, fun_c, lower, upper):    
        self.name = name 
        self.bounds = Bounds(lower, upper)
        self.fun = python_fun(fun_c, self.bounds)

# for windows compatibility. Linux can pickle c pointers, windows can not
astro_map = {  
    "messengerfullC": libgtoplib.messengerfullC,
    "messengerC": libgtoplib.messengerC,
    "gtoc1C": libgtoplib.gtoc1C,
    "cassini1C": libgtoplib.cassini1C,
    "cassini2C": libgtoplib.cassini2C,
    "rosettaC": libgtoplib.rosettaC,
    "sagasC": libgtoplib.sagasC,
    "tandemC": libgtoplib.tandemC,
    "tandemCu": libgtoplib.tandemCu
    }

for func in astro_map:
    astro_map[func].argtypes = [ct.c_int, ct.POINTER(ct.c_double)]           
    astro_map[func].restype = ct.c_double           

class MessFull(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_full/ """
    def __init__(self):    
        Astrofun.__init__(self, 'messenger full', "messengerfullC", 
                           [1900.0, 3.0,    0.0, 0.0,  100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  1.1, 1.1, 1.05, 1.05, 1.05,  -math.pi, -math.pi, -math.pi, -math.pi, -math.pi],
                           [2200.0, 4.05, 1.0, 1.0,  500.0, 500.0, 500.0, 500.0, 500.0, 550.0,  0.99, 0.99, 0.99, 0.99, 0.99, 0.99,  6.0,   6.0,    6.0,    6.0,    6.0,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi]
        )
     
class Messenger(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_reduced/ """

    def __init__(self):    
        Astrofun.__init__(self, 'messenger reduced', "messengerC", 
                           [1000.,1.,0.,0.,200.,30.,30.,30.,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-math.pi,-math.pi,-math.pi],
                           [4000.,5.,1.,1.,400.,400.,400.,400.,0.99,0.99,0.99,0.99,6,6,6,math.pi,math.pi,math.pi]      
        )
    
class Gtoc1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/gtoc1/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'GTOC1', "gtoc1C", 
                           [3000.,14.,14.,14.,14.,100.,366.,300.],
                           [10000.,2000.,2000.,2000.,2000.,9000.,9000.,9000.]       
       )

class Cassini1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini1/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Cassini1', "cassini1C", 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )

class Cassini2(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini2/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Cassini2', "cassini2C", 
            [-1000,3,0,0,100,100,30,400,800,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7, -math.pi, -math.pi, -math.pi, -math.pi],
            [0,5,1,1,400,500,300,1600,2200,0.9,0.9,0.9,0.9,0.9,6,6,6.5,291,math.pi,  math.pi,  math.pi,  math.pi]
        )

class Rosetta(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/rosetta/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Rosetta', "rosettaC", 
            [1460,3,0,0,300,150,150,300,700,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.05,1.05, -math.pi, -math.pi, -math.pi, -math.pi],
            [1825,5,1,1,500,800,800,800,1850,0.9,0.9,0.9,0.9,0.9,9,9,9,9,math.pi,  math.pi,  math.pi,  math.pi]
        )

class Sagas(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/sagas/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Sagas', "sagasC", 
            [7000,0,0,0,50,300,0.01,0.01,1.05,8, -math.pi, -math.pi],
            [9100,7,1,1,2000,2000,0.9,0.9,7,500, math.pi,  math.pi]
        )

class Tandem(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/tandem/ """
    def __init__(self, i, constrained=True):   
        self.name = ('Tandem ' if constrained else 'Tandem unconstrained ') + str(i+1)
        self.cfun = "tandemC" if constrained else "tandemCu"
        self.fun = self.tandem
        self.bounds = Bounds([5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi], 
                             [9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99, 0.99, 10, 10, 10, math.pi,  math.pi,  math.pi])
        self.seqs = [[3,2,2,2,6],[3,2,2,3,6],[3,2,2,4,6],[3,2,2,5,6],[3,2,3,2,6],
                [3,2,3,3,6],[3,2,3,4,6],[3,2,3,5,6],[3,2,4,2,6],[3,2,4,3,6],
                [3,2,4,4,6],[3,2,4,5,6],[3,3,2,2,6],[3,3,2,3,6],[3,3,2,4,6],
                [3,3,2,5,6],[3,3,3,2,6],[3,3,3,3,6],[3,3,3,4,6],[3,3,3,5,6],
                [3,3,4,2,6],[3,3,4,3,6],[3,3,4,4,6],[3,3,4,5,6]]
        self.seq = self.seqs[i]
        
    def tandem(self, x):
        n = len(x)
        array_type = ct.c_double * n   
        ints_type = ct.c_int * 5   
        fun_c = astro_map[self.cfun]      
        fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)]
        try: # function is only defined inside bounds
            x = np.asarray(x).clip(self.bounds.lb, self.bounds.ub)
            val = fun_c(n, array_type(*x), ints_type(*self.seq))
            if not math.isfinite(val):
                val = 1E10
        except Exception as ex:
            val = 1E10
        return val
  
class python_fun(object):
    
    def __init__(self, cfun, bounds):
        self.cfun = cfun
        self.bounds = bounds
    
    def __call__(self, x):
        fun_c = astro_map[self.cfun]      
        n = len(x)
        array_type = ct.c_double * n   
        try: # function is only defined inside bounds
            # x = np.array(x).clip(self.bounds.lb, self.bounds.ub)
            val = float(fun_c(n, array_type(*x)))
            if not math.isfinite(val):
                val = 1E10
        except Exception as ex:
            val = 1E10
        return val
