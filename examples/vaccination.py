# Adapted from:
#
# *Modeling and Simulation in Python*
# 
# Copyright 2021 Allen Downey
# 
# https://github.com/AllenDowney/ModSimPy/blob/master/examples/plague.ipynb
# 
# License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International]
#          (https://creativecommons.org/licenses/by-nc-sa/4.0/)
# Only non-commercial use permitted!
# 
# fcmaes adaption:
#
# Copyright 2022 Dietmar Wolz

import matplotlib.pyplot as plt
import numpy as np    
from numpy import linspace, exp
import pandas as pd    
from numba import njit, numba
import sys    

@njit(fastmath=True)
def fast_update(s, i, r, beta, gamma):
    infected = beta * i * s    
    recovered = gamma * i       
    return s - infected, i + infected - recovered, r + recovered

@njit(fastmath=True)
def fast_simulate(s, i, r, t_end, beta, gamma):
    sa = np.empty(t_end+1)
    ia = np.empty(t_end+1)
    ra = np.empty(t_end+1)
    sa[0] = s
    ia[0] = i
    ra[0] = r
    for t in range(1, t_end+1):
        s, i, r = fast_update(s, i, r, beta, gamma)
        sa[t] = s
        ia[t] = i
        ra[t] = r
    return sa, ia, ra    

@njit(fastmath=True)
def fast_simulate_last(s, i, r, t_end, beta, gamma):
    for t in range(0, t_end):
        s, i, r = fast_update(s, i, r, beta, gamma)
    return s, i, r    

def plot2d(xs, ys, name):
    y1 = ys[:, 0]; y2 = ys[:, 1]; x1 = xs[:, 0]; x2 = xs[:, 1];
    print('size of pareto front', len(y1))
    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()     
    colors = ['#5c9eb7','#f77199','#cf81d2','#4a5e6a','#f45b18']
    plt.xlabel('infection rate')    
    plt.ylabel('spending')
    ax.scatter(y1, x1, s=4, c=colors[1], label='vaccination')
    ax.scatter(y1, x2, s=4, c=colors[2], label='campaign')
    ax.scatter(y1, y2, s=4, c=colors[0], label='sum')
    plt.legend()
    #plt.grid(True, linestyle='dashed')
    major_xticks = np.linspace(0.05, 0.5, 10)
    minor_xticks = np.linspace(0.05, 0.5, 46)
    major_yticks = np.linspace(0, 2400, 13)
    minor_yticks = np.linspace(0, 2400, 61)

    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

def plot3d(xs, ys, name):
    x = ys[:, 0]; y = ys[:, 1]; z = ys[:, 2]
    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()     
    img = ax.scatter(x, y, s=4, c=z, cmap=plt.hot())
    cbar = fig.colorbar(img)
    plt.xlabel('infection rate')    
    plt.ylabel('spending vaccination')
    cbar.set_label('spending campaign')
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

    import plotly
    import plotly.graph_objs as go
    
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
        xaxis=dict(title="infection rate"),
        yaxis=dict( title="spending vaccination"),
        zaxis=dict( title="spending campaign"),
    ),)
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                         auto_open=True,
                         filename=("3DPlot.html"))
  
def execute(task):
    
    SimpleNamespace = type(sys.implementation)

    class SettableNamespace(SimpleNamespace):
 
        def __init__(self, namespace=None, **kwargs):
            super().__init__()
            if namespace:
                self.__dict__.update(namespace.__dict__)
            self.__dict__.update(kwargs)
  
        def get(self, name, default=None):
            try:
                return self.__getattribute__(name, default)
            except AttributeError:
                return default
    
        def set(self, **variables):
            new = copy(self)
            new.__dict__.update(variables)
            return new

    class System(SettableNamespace):
        pass
      
    class Params(SettableNamespace):
        pass
    
    def State(**variables):
        return pd.Series(variables, name='state')

    def TimeSeries(*args, **kwargs):
        if args or kwargs:
            series = pd.Series(*args, **kwargs)
        else:
            series = pd.Series([], dtype=np.float64)
    
        series.index.name = 'Time'
        if 'name' not in kwargs:
            series.name = 'Quantity'
        return series
       
    def SweepSeries(*args, **kwargs):
        if args or kwargs:
            series = pd.Series(*args, **kwargs)
        else:
            series = pd.Series([], dtype=np.float64)
    
        series.index.name = 'Parameter'
        if 'name' not in kwargs:
            series.name = 'Metric'
        return series
    
    def make_series(x, y, **options):
        underride(options, name='values')
        if isinstance(y, pd.Series):
            y = y.values
        series = pd.Series(y, index=x, **options)
        series.index.name = 'index'
        return series
    
    def underride(d, **options):
        if d is None:
            d = {}   
        for key, val in options.items():
            d.setdefault(key, val)
        return d

    def decorate(**options):
        ax = plt.gca()
        ax.set(**options)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)
        plt.tight_layout()
        
    def linrange(start, stop=None, step=1, **options):
        if stop is None:
            stop = start
            start = 0
        n = int(round((stop-start) / step))
        return linspace(start, stop, n+1, **options)

    def TimeFrame(*args, **kwargs):
        underride(kwargs, dtype=float)
        return pd.DataFrame(*args, **kwargs)

    def make_system(beta, gamma):
        init = State(s=89, i=1, r=0)
        init /= init.sum()
        return System(init=init, t_end=7*14,
                      beta=beta, gamma=gamma)
    
    def update_func(t, state, system):
        s, i, r = state.s, state.i, state.r
        infected = system.beta * i * s    
        recovered = system.gamma * i
        s -= infected
        i += infected - recovered
        r += recovered        
        return State(s=s, i=i, r=r)

    def plot_results(S, I, R):
        S.plot(style='--', label='Susceptible')
        I.plot(style='-', label='Infected')
        R.plot(style=':', label='Resistant')
        decorate(xlabel='Time (days)',
                 ylabel='Fraction of population')

    def run_simulation(system, update_func):
        frame = TimeFrame(columns=system.init.index)
        frame.loc[0] = state = system.init

        # original version
        # for t in range(0, system.t_end):
        #     frame.loc[t+1] = update_func(t, frame.loc[t], system)

        # numba based version
        s, i, r = fast_simulate_last(state.s, state.i, state.r, 
                system.t_end, system.beta, system.gamma)
        frame.loc[system.t_end] = State(s=s, i=i, r=r)
        
        return frame    
    
    def add_immunization(system, fraction):
        system.init.s -= fraction
        system.init.r += fraction
    
    tc = 3             # time between contacts in days 
    tr = 4             # recovery time in days
    
    beta = 1 / tc      # contact rate in per day
    gamma = 1 / tr     # recovery rate in per day
    
    system = make_system(beta, gamma)
    
    def calc_total_infected(results, system):
        s_0 = results.s[0]
        s_end = results.s[system.t_end]
        return s_0 - s_end
        
    def logistic(x, A=0, B=1, C=1, M=0, K=1, Q=1, nu=1):
        exponent = -B * (x - M)
        denom = C + Q * exp(exponent)
        return A + (K-A) / denom ** (1/nu)

    spending = linspace(0, 1200, 21)

    def compute_factor(spending):
        return logistic(spending, M=500, K=0.2, B=0.01)

    def add_hand_washing(system, spending):
        factor = compute_factor(spending)
        system.beta *= (1 - factor)
        
    num_students = 90
    budget = 1200
    price_per_dose = 100
    max_doses = int(budget / price_per_dose)
    
    dose_array = linrange(max_doses)
    
    def sweep_doses(dose_array):
        sweep = SweepSeries()
        
        for doses in dose_array:
            fraction = doses / num_students
            spending = budget - doses * price_per_dose
            
            system = make_system(beta, gamma)
            add_immunization(system, fraction)
            add_hand_washing(system, spending)
            
            results = run_simulation(system, update_func)
            sweep[doses] = calc_total_infected(results, system)
    
        return sweep
    
    from fcmaes.optimizer import Bite_cpp, De_cpp, Cma_cpp, Crfmnes_cpp, de_cma, wrapper
    from fcmaes import retry, mode, modecpp, de, bitecpp, decpp, crfmnes, crfmnescpp, cmaes, moretry
    from scipy.optimize import Bounds

    class fcmaes_problem():
        
        def __init__(self):
            self.dim = 2
            self.bounds = Bounds([0]*self.dim, [budget]*self.dim)     
            
        def simulate(self, budged_doses, spending):     
            doses = budged_doses / price_per_dose
            fraction = doses / num_students         
            system = make_system(beta, gamma)
            add_immunization(system, fraction)
            add_hand_washing(system, spending)          
            results = run_simulation(system, update_func)
            return calc_total_infected(results, system)

        def fitness2(self, x):
            budged_doses = x[0]
            spending = x[1]                        
            infected = self.simulate(budged_doses, spending)
            return [infected, budged_doses + spending]

        def fitness3(self, x):
            budged_doses = x[0]
            spending = x[1]                        
            infected = self.simulate(budged_doses, spending)
            return [infected, budged_doses, spending]
        
    problem = fcmaes_problem()    
    
    if task == 'vaccine2dmode':
        xs, ys = mode.minimize(mode.wrapper(problem.fitness2, 2, interval=1000), 2, 
                                    0, problem.bounds, popsize = 256, max_evaluations = 25600, 
                                      nsga_update=False, workers=8)
        xs, ys = moretry.pareto(xs, ys)
        plot2d(xs, ys, task)
    elif task == 'vaccine2dretry':
        xs, ys = modecpp.retry(mode.wrapper(problem.fitness2, 2, interval=1000), 2, 
                   0, problem.bounds, popsize = 128, max_evaluations = 12800, 
                   nsga_update=False, num_retries = 64, workers=32)
        plot2d(xs, ys, task)
    elif task == 'vaccine3dmode':
        xs, ys = mode.minimize(mode.wrapper(problem.fitness3, 3, interval=1000), 3, 
                    0, problem.bounds, popsize = 512, max_evaluations = 25600, 
                    nsga_update=False, workers=8)
        xs, ys = moretry.pareto(xs, ys)
        plot3d(xs, ys, task)
    elif task == 'vaccine3dretry':   
        xs, ys = modecpp.retry(mode.wrapper(problem.fitness3, 3, interval=1000), 3, 
                    0, problem.bounds, popsize = 256, max_evaluations = 25600, 
                    nsga_update=False, workers=32)
        plot3d(xs, ys, task)
            
if __name__ == '__main__':
    # execute('vaccine2dmode')
    execute('vaccine2dretry')
    # execute('vaccine3dmode')
    # execute('vaccine3dretry')


