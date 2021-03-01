# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and is extended here by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# Tested with Anaconda 2020.11 https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

from numba import njit
from skimage.metrics import mean_squared_error
import numpy as np
from scipy import fft
from PIL import Image
import multiprocessing as mp
import math
import ctypes as ct
from scipy.optimize import Bounds
from fcmaes import de, cmaes, retry
from fcmaes.optimizer import de_cma_py, Cma_python, De_python, De_cpp, logger
from time import time

@njit(fastmath=True)
def _get_observed(n_points, im_ft, chromosome):
    r, c = im_ft.shape
    l = 0.01

    x, y = chromosome[:n_points], chromosome[n_points:]

    lx = (np.expand_dims(x, -1) - x).ravel()
    ly = (np.expand_dims(y, -1) - y).ravel()

    theta = np.linspace(0, 2*np.pi, 10000)
    cos_theta = np.cos(np.expand_dims(theta, -1))
    sin_theta = np.sin(np.expand_dims(theta, -1))

    full_re_u = ((lx *  cos_theta + ly * sin_theta ) / l).astype(np.int32)
    full_re_v = ((lx * -sin_theta + ly * cos_theta ) / l).astype(np.int32)    

    full_re_uimg = (full_re_u * r / 2**2.5 * l).astype(np.int32)
    full_re_vimg = (full_re_v * r / 2**2.5 * l).astype(np.int32)

    obs_uv_matrix = np.zeros((r, c), dtype=np.int32)
    for x, y in zip(full_re_uimg.flat, full_re_vimg.flat):
        obs_uv_matrix[x][y] = 1

    return im_ft * obs_uv_matrix
 
best = mp.RawValue(ct.c_double, math.inf) 
count = mp.RawValue(ct.c_int, 0) 
t0 = time()

class Interferometry():
    def __init__(self, number_points, image, image_size):
        self.number_points = number_points
        self.image = image

        assert((image_size & (image_size-1) == 0) and image_size != 0)
        self.image_size_log2 = np.log2(image_size)
        self.image_size = image_size

        img = Image.open(self.image)
        img = img.resize((self.image_size,self.image_size))

        self.im_numpy = np.asarray(img)
        self.im_fft = fft.fft2(self.im_numpy)

    def get_bounds(self):
        return ([-1.0]*self.number_points*2, [1.0]*self.number_points*2)

    def fitness(self, x):
        count.value += 1
        observed = _get_observed(self.number_points, self.im_fft, x)
        im_reconstruct = fft.ifft2(observed).real
        val = mean_squared_error(self.im_numpy, im_reconstruct)
        if val < best.value:
            best.value = val
            print(str(count.value) + ' fval = ' + str(val) + 
                  " t = " + str(round(time() - t0)) + " s" + " x = " + ", ".join(str(xi) for xi in x))
        return val
    
def parallel_de():   
    global best,count,t0
    udp = Interferometry(11, './img/orion.jpg', 512)
    bounds = Bounds(udp.get_bounds()[0], udp.get_bounds()[1])
    for i in range(10):
        best = mp.RawValue(ct.c_double, math.inf) 
        count = mp.RawValue(ct.c_int, 0) 
        t0 = time()
        print('interferometer de parallel function evaluation run ' + str(i))
        
        # Python Differential Evolution implementation, uses ask/tell for parallel function evaluation.
        ret = de.minimize(udp.fitness, bounds=bounds, workers=6, popsize=31, max_evaluations=50000)
        
        # Python CMAES implementation, uses ask/tell for parallel function evaluation.
        #ret = cmaes.minimize(udp.fitness, bounds=bounds, workers=6, popsize=31, max_evaluations=50000)
        
        # parallel retry. We can use the C++ version of CMA since parallelization is at the retry level. 
        #ret = retry.minimize(udp.fitness, bounds=bounds, optimizer=Cma_cpp(50000, popsize=31), workers=6)
        
        print("best result is " + str(ret.fun) + ' x = ' + ", ".join(str(x) for x in ret.x))

if __name__ == '__main__':
    parallel_de()
    pass