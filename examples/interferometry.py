# This code was posted on https://gitter.im/pagmo2/Lobby by 
# Markus Märtens @CoolRunning and extended by a 
# fcmaes parallel differential evolution solver for comparison with the pagmo island concept.
# requires oagmo (pip install pagmo) for the comparison. Tested with Anaconda 2020.11 
# https://repo.anaconda.com/archive/ using Python 3.8 on Linux
# The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg

from numba import njit
from skimage.metrics import mean_squared_error
import numpy as np
from scipy import fft
from PIL import Image
import multiprocessing as mp
import math
import ctypes as ct
import pygmo as pg

from fcmaes import de
from fcmaes.optimizer import single_objective, eprint
from time import time

@njit
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

# Sharing values between processes doesn't work well with windows, they are copied instead.
# But this is only to monitor progress, parallel function evaluation itself works fine. 
# Nevertheless on windows it is better to use the linux subsystem. 
best = mp.RawValue(ct.c_double, math.inf) 
count = mp.RawValue(ct.c_int, 0) 
t0 = time()

class Interferometry():
    def __init__(self, number_points, image, image_size):
        self.number_points = number_points
        self.image = image

        #We make sure that it is a power of 2
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
        val = math.inf
        try:
            observed = _get_observed(self.number_points, self.im_fft, x)
            im_reconstruct = fft.ifft2(observed).real
            val = (mean_squared_error(self.im_numpy, im_reconstruct),)
            if val[0] < best.value:
                best.value = val[0]
                print(str(count.value) + ' fval = ' + str(val[0]) + 
                      " t = " + str(round(time() - t0)) + " s" + " x = " + ", ".join(str(xi) for xi in x))
        except Exception as ex:
            eprint('error in fitness ' + str(ex))
        
        return val

def archipelago():    
    uda = pg.sga(gen = 1000)
    udp = Interferometry(11, './img/orion.jpg', 512)
    # instantiate an unconnected archipelago
    archi = pg.archipelago(t = pg.topologies.unconnected())
    
    for _ in range(4):
        alg = pg.algorithm(uda)
        alg.set_verbosity(1)
    
        udp = Interferometry(11, './img/orion.jpg', 512)
        prob = pg.problem(udp)
        pop = pg.population(prob, 20)
    
        isl = pg.island(algo=alg, pop=pop)
        archi.push_back(isl)   
    
    t = time()
    archi.evolve()
    archi.wait_check()
    print(f'archi: {time() - t:0.3f}s')
    
def parallel_de():   
    global best,count,t0
    udp = Interferometry(11, './img/orion.jpg', 512)
    for i in range(10):
        best = mp.RawValue(ct.c_double, math.inf) 
        count = mp.RawValue(ct.c_int, 0) 
        t0 = time()
        fprob = single_objective(pg.problem(udp))
        print('interferometer de parallel function evaluation run ' + str(i))
        ret = de.minimize(fprob.fun, bounds=fprob.bounds, workers=32, popsize=32, max_evaluations=100000)
        print("best result is " + str(ret.fun) + ' x = ' + ", ".join(str(x) for x in ret.x))

if __name__ == '__main__':
    parallel_de()
    #archipelago()
    pass