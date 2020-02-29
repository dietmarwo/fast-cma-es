# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import math
import os
import ctypes as ct
from scipy.optimize import Bounds

basepath = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith('linux'):
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libgtoplib.so')    
else:
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libgtoplib.dll')

class Astrofun(object):
    """Provides access to ESAs GTOP optimization test functions."""
    def __init__(self, name, fun_c, lower, upper):    
        self.name = name 
        self.fun_c = fun_c
        self.fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double)] 
        self.fun_c.restype = ct.c_double           
        self.fun = _python_fun(self.fun_c)
        self.bounds = Bounds(lower, upper)

class MessFull(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_full/ """
    def __init__(self):    
        Astrofun.__init__(self, 'messenger full', libgtoplib.messengerfullC, 
                           [1900.0, 3.0,    0.0, 0.0,  100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  1.1, 1.1, 1.05, 1.05, 1.05,  -math.pi, -math.pi, -math.pi, -math.pi, -math.pi],
                           [2200.0, 4.05, 1.0, 1.0,  500.0, 500.0, 500.0, 500.0, 500.0, 550.0,  0.99, 0.99, 0.99, 0.99, 0.99, 0.99,  6.0,   6.0,    6.0,    6.0,    6.0,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi]
        )
     
class Messenger(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_reduced/ """

    def __init__(self):    
        Astrofun.__init__(self, 'messenger reduced', libgtoplib.messengerC, 
                           [1000.,1.,0.,0.,200.,30.,30.,30.,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-math.pi,-math.pi,-math.pi],
                           [4000.,5.,1.,1.,400.,400.,400.,400.,0.99,0.99,0.99,0.99,6,6,6,math.pi,math.pi,math.pi]      
        )
    
class Gtoc1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/gtoc1/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'GTOC1', libgtoplib.gtoc1C, 
                           [3000.,14.,14.,14.,14.,100.,366.,300.],
                           [10000.,2000.,2000.,2000.,2000.,9000.,9000.,9000.]       
       )

class Cassini1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini1/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Cassini1', libgtoplib.cassini1C, 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )

class Cassini2(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini2/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Cassini2', libgtoplib.cassini2C, 
            [-1000,3,0,0,100,100,30,400,800,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7, -math.pi, -math.pi, -math.pi, -math.pi],
            [0,5,1,1,400,500,300,1600,2200,0.9,0.9,0.9,0.9,0.9,6,6,6.5,291,math.pi,  math.pi,  math.pi,  math.pi]
        )

class Rosetta(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/rosetta/ """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Rosetta', libgtoplib.rosettaC, 
            [1460,3,0,0,300,150,150,300,700,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.05,1.05, -math.pi, -math.pi, -math.pi, -math.pi],
            [1825,5,1,1,500,800,800,800,1850,0.9,0.9,0.9,0.9,0.9,9,9,9,9,math.pi,  math.pi,  math.pi,  math.pi]
        )
    
def _python_fun(cfun):
    return lambda x : _call_c(cfun, x)

def _call_c(cfun, x):
    n = len(x)
    array_type = ct.c_double * n   
    try:
        val = float(cfun(n, array_type(*x)))
    except:
        val = sys.float_info.max
    return val
