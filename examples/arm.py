# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""
Basic example for quality diversity optimization:

A planar robotic arm consists of dim+1 = 21 segments with dim=20 joints. 

For each reachable position minimize either:

- the standard deviation of the joint-angles
- the sum of the joint-angles
- the maximal joint-angle

Since all results are stored in the ND archive finally we know for all target positions
how to adjust all the joint-angles to meet the objective without any further optimization. 

On a modern 16 core CPU the whole optimization takes about 30 seconds. 

Play around with the parameters of the experiment:

- select different opt_params (optimization algorithms + their configurations)
- select different objectives
- change the dimension dim - the number of joint-angles

and check the resulting plot.  
"""

from fcmaes import diversifier, mapelites
import numpy as np
from scipy.optimize import Bounds

import ctypes as ct
import multiprocessing as mp 

def forward_kinematics(q):
    """
    Calculates links positions given joint angles
    Parameters
    ----------
    q : numpy.ndarray
        (n_links,) array of angles in radians
        
    adapted from https://github.com/tsitsimis/planar-robot/blob/master/planarobot/planar_arm.py
    """
    n_links = len(q) + 1
    base_pos = (0,0)
    links = np.full(n_links, 1)
    pos = np.zeros((n_links, 2))
    pos[0, :] = base_pos + np.array([[links[0] * np.cos(q[0]), links[0] * np.sin(q[0])]])
    for i in range(1, n_links):
        delta_pos = np.array([links[i] * np.cos(np.sum(q[0:i+1])),
                              links[i] * np.sin(np.sum(q[0:i+1]))])
        pos[i, :] = pos[i - 1, :] + delta_pos
    return pos

class fitness(object):

    def __init__(self, dim):
        self.dim = dim 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.bounds = Bounds([-np.pi]*dim, [np.pi]*dim)
        self.qd_dim = 2
        self.qd_bounds = Bounds([-21, -21], [21, 21])

    def __call__(self, X):
        pos = forward_kinematics(X)
        #y = np.amax(np.abs(X))
        #y = np.sum(np.abs(X))
        y = np.std(X)
        b = pos[-1]
        if y < self.best_y.value:
            self.best_y.value = y
            print(f'{y:.3f} { list(b) }')            
        return y, b  
     
def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    x = ys[:, 0]; y = ys[:, 1]; z = ys[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot()     
    img = ax.scatter(x, y, s=4, c=z, cmap='rainbow')
    cbar = fig.colorbar(img)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='dashed')
    cbar.set_label(zlabel)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

def plot_archive(problem, archive = None):
    name = 'arm_nd'
    if archive is None:
        archive = mapelites.load_archive(name, problem.bounds, problem.qd_bounds)
    si = archive.argsort()
    ysp = []
    beh = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    lb = problem.qd_bounds.lb
    ub = problem.qd_bounds.ub
    for i in range(len(si)):
        # if ys[i] < 5.0: # use for y = np.sum(np.abs(X))
        if ys[i] < 0.2: # use for y = np.std(X)
            b = beh[i]
            if np.any(np.greater(b, ub)) or np.any(np.greater(lb, b)):
                continue
            ysp.append([b[0], b[1], ys[i]])
    ysp = np.array(ysp)
    plot3d(ysp, name, 'x', 'y', 'objective')
                
def optimize_qd(dim):
    problem = fitness(dim)
    name = 'arm_nd'
    opt_params0 = {'solver':'elites', 'popsize':512}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    opt_params2 = {'solver':'DE_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    opt_params3 = {'solver':'CMA_CPP', 'max_evals':4000, 'popsize':32, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem, problem.qd_dim, interval=200000, save_interval=12000000), 
         problem.bounds, problem.qd_bounds, opt_params=[opt_params0, opt_params1], max_evals=3000000) 
    print('final archive:', archive.info())
    archive.save(name)
    plot_archive(problem, archive)
    
if __name__ == '__main__':    
    dim = 20
    # apply a QD algorithm
    optimize_qd(dim)
    # plot the result
    plot_archive(fitness(dim))

 
