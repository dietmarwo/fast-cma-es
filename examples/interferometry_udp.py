# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and extended by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# requires oagmo (pip install pagmo) for the comparison. Tested with Anaconda 2020.11 
# https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

import math
from time import time

from matplotlib import pyplot
from numba import njit
from scipy import fft
from skimage.metrics import mean_squared_error
from skimage.transform import resize

import ctypes as ct
import multiprocessing as mp
import numpy as np

@njit(fastmath=True)
def _get_observed(n_points, im_ft, chromosome):
    r, c = im_ft.shape
    l = 0.01
    x, y = chromosome[:n_points], chromosome[n_points:]
    
    lx = (np.expand_dims(x, -1) - x).ravel()
    ly = (np.expand_dims(y, -1) - y).ravel()

    theta = np.linspace(0, 2*np.pi, 10000)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    obs_uv_matrix = np.zeros((r, c), dtype=np.int32)
         
    for i in range(10000):
        for j in range(len(lx)):         
            full_re_u = math.floor((lx[j] *  cos_theta[i] + ly[j] * sin_theta[i] ) / l)
            full_re_v = math.floor((lx[j] * -sin_theta[i] + ly[j] * cos_theta[i] ) / l)   
            x = math.floor(full_re_u * r / 2**2.5 * l)
            y = math.floor(full_re_v * r / 2**2.5 * l)
            obs_uv_matrix[x][y] = 1
            
    return im_ft * obs_uv_matrix

best = mp.RawValue(ct.c_double, math.inf) 
count = mp.RawValue(ct.c_int, 0) 
t0 = time()
monitor = mp.Lock()

class Interferometry():
    def __init__(self, number_points, image, image_size):
        self.number_points = number_points
        self.image = image

        #We make sure that it is a power of 2
        assert((image_size & (image_size-1) == 0) and image_size != 0)
        self.image_size_log2 = np.log2(image_size)
        self.image_size = image_size

        img = pyplot.imread(self.image)
        img = resize(img, (self.image_size,self.image_size), preserve_range=True, anti_aliasing=True)

        self.im_numpy = np.asarray(img)
        self.im_fft = fft.fft2(self.im_numpy)

    def get_bounds(self):
        return ([-1.0]*self.number_points*2, [1.0]*self.number_points*2)

    def fitness(self, x):
        observed = _get_observed(self.number_points, self.im_fft, x)        
        im_reconstruct = fft.ifft2(observed).real
        val = (mean_squared_error(self.im_numpy, im_reconstruct),)
        with monitor:
            count.value += 1
            if val[0] < best.value:
                best.value = val[0]
                print(str(count.value) + ' fval = ' + str(val[0]) + 
                      " t = " + str(round(1000*(time() - t0))) + " ms" + " x = " + ", ".join(str(xi) for xi in x))
        return val
    