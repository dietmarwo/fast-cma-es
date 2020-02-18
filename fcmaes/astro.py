'''
Created on Jan 24, 2020

@author: Dietmar Wolz
'''

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
        self.fun = python_fun(self.fun_c)
        self.bounds = Bounds(lower, upper)

class MessFull(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_full.html """
    def __init__(self):    
        Astrofun.__init__(self, 'messenger full', libgtoplib.messengerfullC, 
                           [1900.0, 3.0,    0.0, 0.0,  100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  1.1, 1.1, 1.05, 1.05, 1.05,  -math.pi, -math.pi, -math.pi, -math.pi, -math.pi],
                           [2200.0, 4.05, 1.0, 1.0,  500.0, 500.0, 500.0, 500.0, 500.0, 550.0,  0.99, 0.99, 0.99, 0.99, 0.99, 0.99,  6.0,   6.0,    6.0,    6.0,    6.0,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi]
        )
     
class Messenger(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/messenger_reduced.html """

    def __init__(self):    
        Astrofun.__init__(self, 'messenger reduced', libgtoplib.messengerC, 
                           [1000.,1.,0.,0.,200.,30.,30.,30.,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-math.pi,-math.pi,-math.pi],
                           [4000.,5.,1.,1.,400.,400.,400.,400.,0.99,0.99,0.99,0.99,6,6,6,math.pi,math.pi,math.pi]      
        )
    
class Gtoc1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/gtoc1.html """
    
    def __init__(self):    
        Astrofun.__init__(self, 'GTOC1', libgtoplib.gtoc1C, 
                           [3000.,14.,14.,14.,14.,100.,366.,300.],
                           [10000.,2000.,2000.,2000.,2000.,9000.,9000.,9000.]       
       )

class Cassini1(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini1.html """
    
    def __init__(self):    
        Astrofun.__init__(self, 'Cassini1', libgtoplib.cassini1C, 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )
    
def python_fun(cfun):
    return lambda x : call_c(cfun, x)

def call_c(cfun, x):
    n = len(x)
    array_type = ct.c_double * n   
    try:
        val = float(cfun(n, array_type(*x)))
    except:
        val = sys.float_info.max
    return val
