# Adapted from https://github.com/slds-lmu/qdo_yahpo for application of fcmaes
#
# See "A Collection of Quality Diversity Optimization Problems Derived 
# from Hyperparameter Optimization of Machine Learning Models" https://arxiv.org/abs/2204.14061
# 
# Differences to https://github.com/slds-lmu/qdo_yahpo:
# - No statistics, just applies the Hyperparameter Optimization and ranger benchmark functions
# - Parallelism is handled by the optimizer
# - Configured optimizer is similar to "mixed" in qdo_yahpo, but
#     * Uses CR-FM-NES instead of CMA-ES emitter
#     * CR-FM-NES is initialized with a random solution instead of a niche elite
#     * Uses Voronoi tesselation (see CVT MAP-Elites https://arxiv.org/abs/1610.05729)
#     * Map-Elites is executed in parallel with CR-FM-NES sharing the same QD-archive
#     * Instead of gaussian distribution Map-Elites uses simulated binary crossover + mutation as NSGAII
# 
# Results cannot be directly compared since different tesselation is used (Grid/Voronoi). 
# 
# Requires yahpo_gym, please follow the installation instuctions at 
# https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym

from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.iaml
import numpy as np
import pandas as pd
from fcmaes import diversifier, mapelites
from scipy.optimize import Bounds

def opt_ranger_hardware(task_id, max_evals = 100000):
    bench = benchmark_set.BenchmarkSet("iaml_ranger", check = False)
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()

    params = ["num.trees", "mtry.ratio", "min.node.size", "sample.fraction"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    lower = [bound[0] for bound in bounds]
    upper = [bound[1] for bound in bounds]
    
    if task_id == "41146":
        desc_bounds = Bounds((1., 0.19), (200., 4.5)) 
    elif task_id == "40981":
        desc_bounds = Bounds((1., 0.10), (40., 0.65)) 
    elif task_id == "1489":
        desc_bounds = Bounds((1., 0.19), (200., 4.5)) 
    elif task_id == "1067":
        desc_bounds = Bounds((1., 0.13), (78., 1.55)) 
    
    dim = len(bounds)
    desc_dim = 2
    bounds = Bounds([0]*dim, [1]*dim)
    
    def qd_fun(x): # Only a single fitness is needed. Parallelism is handled by the optimizer.
        yd = evaluate_ranger_([x], ["rammodel", "timepredict"], params, bench, task_id, lower, upper)
        return yd[0][0], yd[0][1:]
    
    niche_num = 4000
    arch = None                   
    opt_params0 = {'solver':'elites', 'popsize':16}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':200, 'popsize':16, 'stall_criterion':3}
    
    archive = diversifier.minimize(
         mapelites.wrapper(qd_fun, desc_dim, interval=100000), 
         bounds, desc_bounds, opt_params=[opt_params0, opt_params1], max_evals=max_evals, archive = arch,
         niche_num = niche_num, samples_per_niche = 20)
    
    name = "range_hard" + task_id
    print('final archive:', archive.info())
    archive.save(name)
    plot(archive, name)

def opt_ranger_interpretability(task_id, max_evals = 100000):
    bench = benchmark_set.BenchmarkSet("iaml_ranger", check = False)
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()

    params = ["num.trees", "mtry.ratio", "min.node.size", "sample.fraction"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    lower = [bound[0] for bound in bounds]
    upper = [bound[1] for bound in bounds]
    
    if task_id == "41146":
        desc_bounds = Bounds((0., 0.), (20., 1.)) 
    elif task_id == "40981":
        desc_bounds = Bounds((0., 0.), (14., 1.)) 
    elif task_id == "1489":
        desc_bounds = Bounds((0., 0.), (5., 1.)) 
    elif task_id == "1067":
        desc_bounds = Bounds((0., 0.), (21., 1.))   
           
    dim = len(bounds)
    desc_dim = 2
    bounds = Bounds([0]*dim, [1]*dim)
    
    def qd_fun(x): # Only a single fitness is needed. Parallelism is handled by the optimizer.
        yd = evaluate_ranger_([x], ["nf", "ias"], params, bench, task_id, lower, upper)
        return yd[0][0], yd[0][1:]
    
    niche_num = 4000
    arch = None                 
    opt_params0 = {'solver':'elites', 'popsize':16}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':200, 'popsize':16, 'stall_criterion':3}
    
    archive = diversifier.minimize(
         mapelites.wrapper(qd_fun, desc_dim, interval=100000), 
         bounds, desc_bounds, opt_params=[opt_params0, opt_params1], max_evals=max_evals, archive = arch,
         niche_num = niche_num, samples_per_niche = 20)
    
    name = "range" + task_id
    print('final archive:', archive.info())
    archive.save(name)
    plot(archive, name)

def opt_xgboost(task_id, max_evals = 100000):
    bench = benchmark_set.BenchmarkSet("iaml_xgboost", check = False)  # we disable input checking of parameters for speed up
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()
    params = ["alpha", "lambda", "nrounds", "subsample", "colsample_bylevel", "colsample_bytree", "eta", "gamma", "max_depth", "min_child_weight"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    defaults = [0.001, 0.001, 1000., 1., 1., 1., 0.3, 0.001, 6., 3.]
    for i in range(len(params)):
        if search_space.get_hyperparameter(params[i]).log:
            bounds[i] = np.log(bounds[i])
            defaults[i] = np.log(defaults[i])
    lower = [bound[0] for bound in bounds]
    upper = [bound[1] for bound in bounds]

    if task_id == "41146":
        desc_bounds = Bounds((0., 0.), (20., 1.)) 
    elif task_id == "40981":
        desc_bounds = Bounds((0., 0.), (14., 1.)) 
    elif task_id == "1489":
        desc_bounds = Bounds((0., 0.), (5., 1.)) 
    elif task_id == "1067":
        desc_bounds = Bounds((0., 0.), (21., 1.))        
    desc_dim = 2
    
    dim = len(defaults)
    bounds = Bounds([0]*dim, [1]*dim)
    
    def qd_fun(x):
        yd = evaluate_xgboost_([x], ["nf", "ias"], params, bench, task_id, lower, upper)
        return yd[0][0], yd[0][1:]
    
    niche_num = 4000
    arch = None                  
    opt_params0 = {'solver':'elites', 'popsize':16}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':200, 'popsize':16, 'stall_criterion':3}
     
    archive = diversifier.minimize(
         mapelites.wrapper(qd_fun, desc_dim, interval=100000), 
         bounds, desc_bounds, opt_params=[opt_params0, opt_params1], max_evals=max_evals, archive = arch,
         niche_num = niche_num, samples_per_niche = 20)
    
    name = "xgboost_opt" + task_id
    print('final archive:', archive.info())
    archive.save(name)
    plot(archive, name)

def evaluate_ranger_(x, features, params, bench, task_id, lower, upper, trafo=True):
    if trafo:
        x = np.array([retrafo_(i, lower, upper) for i in x])
    # YAHPO Gym supports batch predicts if configs are passed as a list of dicts
    config = [dict(zip(params, x[i])) for i in range(x.shape[0])]
    # add the other hyperparameters not part of the search space and respect integers
    for i in range(len(config)):
        config[i].update({"num.trees":round(config[i]["num.trees"]), "min.node.size":round(config[i]["min.node.size"])})
        config[i].update({"task_id":task_id, "trainsize":1, "replace":"TRUE", "respect.unordered.factors":"ignore", "splitrule":"gini"})
    results = bench.objective_function(config)
    targets = ["mmce"]
    targets.extend(features)
    y = pd.DataFrame(results)[targets]
    y[["mmce"]] = 1 - y[["mmce"]]  # pyribs maximizes by default so we turn mmce into acc
    return y.values  # pyribs expects a numpy array as return value

def evaluate_xgboost_(x, features, params, bench, task_id, lower, upper, trafo=True):
    if trafo:
        x = np.array([retrafo_(i, lower, upper) for i in x])
    # YAHPO Gym supports batch predicts if configs are passed as a list of dicts
    config = [dict(zip(params, x[i])) for i in range(x.shape[0])]
    # add the other hyperparameters not part of the search space and respect log trafos and integers
    for i in range(len(config)):
        config[i].update({"alpha":np.exp(config[i]["alpha"]), "lambda":np.exp(config[i]["lambda"]), "eta":np.exp(config[i]["eta"]), "gamma":np.exp(config[i]["gamma"]), "min_child_weight":np.exp(config[i]["min_child_weight"])})
        config[i].update({"nrounds":round(np.exp(config[i]["nrounds"])), "max_depth":round(config[i]["max_depth"])})
        config[i].update({"task_id":task_id, "trainsize":1, "booster":"gbtree"})
    results = bench.objective_function(config)
    targets = ["mmce"]
    targets.extend(features)
    y = pd.DataFrame(results)[targets]
    y[["mmce"]] = 1 - y[["mmce"]]  # pyribs maximizes by default so we turn mmce into acc
    return y.values  # pyribs expects a numpy array as return value

def retrafo_(z, lower, upper):
    n = len(z)
    assert(n == len(lower))
    assert(n == len(upper))
    return [np.max((lower[i], np.min(((z[i] * (upper[i] - lower[i])) + lower[i], upper[i])))) for i in range(n)]

def plot3d(ys, name, xlabel='', ylabel='', zlabel=''):
    import matplotlib.pyplot as plt
    x = ys[:, 0]; y = ys[:, 1]; z = ys[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot()     
    ax.grid()
    img = ax.scatter(x, y, s=4, c=z, cmap='rainbow')
    cbar = fig.colorbar(img)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)
    plt.grid(visible=True)
    cbar.set_label(zlabel)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

def plot(archive, name):
    si = archive.argsort()
    ysp = []
    ds = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    for i in range(len(si)):
        if ys[i] == np.inf: 
            break
        d = ds[i]
        ysp.append([d[0], d[1], ys[i]])
    ysp = np.array(ysp)
    print(len(ysp))
    print(ysp)
    plot3d(ysp, name, 'x', 'y', 'z')
    
if __name__ == '__main__':
    for task_id in ["41146", "40981", "1489", "1067"]:        
        opt_ranger_interpretability(task_id)
        opt_ranger_hardware(task_id)
        opt_xgboost(task_id)
