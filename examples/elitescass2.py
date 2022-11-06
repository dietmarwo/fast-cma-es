'''
Created on Oct 29, 2022

@author: xxx
'''
import numpy as np
from scipy.optimize import Bounds
from fcmaes import mapelites
from fcmaes.astro import Cassini2

def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objs as go
    x = ys[:, 0]; y = ys[:, 1]; z = ys[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot()     
    img = ax.scatter(x, y, s=4, c=z, cmap='rainbow')
    cbar = fig.colorbar(img)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)
    cbar.set_label(zlabel)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)
    
    fig1 = go.Scatter3d(x=x,
                    y=y,
                    z=z,
                    marker=dict(opacity=0.9,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')
    mylayout = go.Layout(scene=dict(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        zaxis=dict(title=zlabel),
    ),)
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                         auto_open=True,
                         filename=(name + ".html"))
def plot_archive(archive, max_dv = 20):
    si = archive.argsort()
    ysp = []
    descriptions = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    for i in range(len(si)):
        desc = descriptions[i]
        ysp.append([desc[0], desc[1], ys[i]])
        if ys[i] > max_dv: break
    ysp = np.array(ysp)
    print(len(ysp))
    print(ysp)
    plot3d(ysp, "cassini_2d", 'time of flight', 'start day', 'delta V / propellant')

def tof(x):
    return sum(x[4:9])

def launch(x):
    return x[0]

class Cassini2_me():
    ''' Map-Elites wrapper for the ESA Cassini2 benchmark problem'''
    
    def __init__(self, prob):
        self.problem = prob
        self.dim = len(prob.bounds.lb)
        self.desc_dim = 2
        self.bounds = prob.bounds
        
        min_tof = tof(prob.bounds.lb)
        max_tof = tof(prob.bounds.ub)
        min_launch = launch(prob.bounds.lb)
        max_launch = launch(prob.bounds.ub)

        self.desc_bounds = Bounds([min_tof, min_launch], [max_tof, max_launch]) 
                        
    def fun(self, x):
        return self.problem.fun(x), np.array([tof(x), launch(x)])

def cma_elite(problem, archive, num=300):    
    ''' applies CMA-ES to the best num niches'''
    si = archive.argsort()
    for i in range(1, num+1):
        try:
            j = si[i]
            print (j, archive.get_count(j))
            print (archive.get_x_mean(j))
            print (archive.get_x_min(j))
            print (archive.get_x_max(j))
            print (list(archive.get_x_stdev(j)))
           
            guess = archive.get_x(j) 
            fun = archive.in_niche_filter(problem.fun, j)
            print (archive.get_y(j), fun(guess))
    
            lb = np.nan_to_num(archive.get_x_min(j), nan=-np.inf)
            ub = np.nan_to_num(archive.get_x_max(j), nan=np.inf) 
            bounds = Bounds(np.maximum(problem.bounds.lb, lb), 
                            np.minimum(problem.bounds.ub, ub)) 
            from fcmaes import retry
            from fcmaes.optimizer import logger, Cma_cpp
            res = retry.minimize(fun, bounds, num_retries=24*8, logger=logger(), 
                           optimizer=Cma_cpp(guess=guess, sdevs=0.001, workers=24)
                           )       
            y, d = problem.fun(res.x) 
            print (j, res.fun, fun(res.x), y, d)
            archive.set(j, [y,d], res.x)
            if i % 50 == 0:
                archive.save("cass2archCma" + str(i))
        except Exception as ex:
            pass
    archive.save("cass2archCma")

niche_num = 4000    

def plot(name):
    problem = Cassini2_me(Cassini2())
    archive = mapelites.load_archive(name, problem.bounds, problem.desc_bounds, niche_num)
    plot_archive(archive)
    
def run_map_elites():
    problem = Cassini2_me(Cassini2())
    name = 'cass2arch'
    archive = None
    #archive = mapelites.load_archive("cass2arch",  problem.bounds, problem.desc_bounds, niche_num)

    me_params = {'generations':100, 'chunk_size':1000}
    cma_params = {'cma_generations':100, 'best_n':200, 'maxiters':1000, 'miniters':200}
    
    fitness =  mapelites.wrapper(problem.fun, problem.desc_dim)

    archive = mapelites.optimize_map_elites(
        fitness, problem.bounds, problem.desc_bounds, niche_num = niche_num,
          min_capacity = 0.1, capacity_reduce = 0.97, iterations = 100, archive = archive, 
          me_params = me_params, cma_params = cma_params)
    archive.save(name)
    me_params['best_n'] = 1000
    for i in range(20):
        archive = mapelites.optimize_map_elites(
            fitness, problem.bounds, problem.desc_bounds, niche_num = niche_num, 
                min_capacity = 0.1, capacity_reduce = 0.97, iterations = 100, archive = archive, 
                me_params = me_params, cma_params = cma_params)
        archive.save(name + '.' + str(i))
    plot(archive)

if __name__ == '__main__':
    
    #run_map_elites()
    plot('cass2arch.3')
    pass