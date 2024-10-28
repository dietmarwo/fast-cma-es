
'''
This example is based on election prediction data from
https://en.wikipedia.org/wiki/The_Keys_to_the_White_House

The prediction model was developed by Allan Lichtman and Russian geophysicist Vladimir Keilis-Borok 
and uses 13 key properties to predict the outcome of an election. 

It can predict correctly almost all past presidential elections in the USA since 1860. 

This example proves, that it cannot reliably predict the 2024 election. 

Lichtmann assigns all his criteria the same weight (1.0) and defines a limit sum, 
which fits well with past elections.

If we could apply different weights and thresholds to Lichtman's criteria, 
it suggests that the model could yield alternative predictions, 
especially for elections where the context varies from those on which the model 
was originally calibrated. This means that while the "13 Keys" may fit historical data well, 
other configurations might fit just as well but lead to different forecasts. 
Essentially, Lichtman’s model may not be the only equally valid model; 
alternative configurations could yield differing, yet valid, predictions, 
implying potential flexibility in interpreting election outcomes.

To prove this statement we have to find weights for the 13 keys and corresponding limits, so that
a) It is predicted that Harris wins 2024
and other weights + limits, so that 
b) It is predicted that Trump wins 2024

Executing test_harris_prediciton() we get:

Harris looses

num correct 42.0 weights [1.684, 1.731, 1.968, 0.47, 2.326, 1.491, 0.104, 0.924, 0.93, 0.516, 0.213, 0.585, 0.057] limit 4.75
num correct 42.0 weights [1.637, 1.552, 1.122, 0.015, 1.434, 0.329, 0.969, 0.697, 0.487, 1.711, 0.024, 1.709, 1.314] limit 5.458
num correct 42.0 weights [1.457, 1.685, 0.888, 0.072, 1.996, 0.115, 0.576, 0.949, 0.95, 1.641, 0.292, 1.186, 1.191] limit 4.915
num correct 42.0 weights [1.483, 1.538, 1.71, 0.039, 1.648, 0.204, 0.512, 0.629, 0.907, 1.21, 0.459, 1.466, 1.196] limit 5.261
num correct 42.0 weights [1.709, 1.71, 1.507, 0.9, 1.072, 0.216, 0.924, 0.93, 1.092, 1.54, 0.258, 0.919, 0.224] limit 5.343
num correct 42.0 weights [1.04, 1.247, 1.462, 0.667, 1.423, 0.357, 0.539, 0.992, 0.762, 1.444, 0.522, 1.203, 1.34] limit 5.122
num correct 42.0 weights [1.636, 2.098, 1.194, 0.455, 1.312, 0.493, 0.383, 0.499, 0.802, 1.237, 0.136, 1.725, 1.031] limit 5.789
num correct 42.0 weights [2.028, 0.921, 1.675, 0.644, 1.532, 0.313, 0.224, 0.36, 0.277, 1.382, 0.995, 1.474, 1.176] limit 5.871
num correct 42.0 weights [2.746, 2.226, 1.448, 0.419, 0.801, 0.137, 1.11, 0.157, 1.799, 1.203, 0.663, 0.011, 0.28] limit 5.056
num correct 42.0 weights [1.375, 1.496, 1.436, 0.252, 1.553, 0.555, 0.807, 0.558, 0.695, 1.817, 0.988, 0.489, 0.98] limit 4.561

Harris wins

num correct 42.0 weights [1.227, 0.999, 1.124, 1.446, 0.937, 0.89, 1.47, 1.059, 0.668, 1.089, 0.419, 0.712, 0.958] limit 5.597
num correct 42.0 weights [1.449, 1.521, 1.416, 0.982, 1.456, 1.202, 0.119, 0.791, 0.847, 0.099, 0.695, 1.111, 1.312] limit 5.328
num correct 42.0 weights [1.598, 0.878, 1.299, 1.021, 1.233, 0.883, 0.927, 1.146, 0.804, 0.956, 1.384, 0.137, 0.734] limit 5.603
num correct 42.0 weights [0.202, 1.353, 0.649, 1.236, 0.516, 1.37, 1.094, 1.305, 1.2, 0.354, 0.932, 1.479, 1.311] limit 5.455
num correct 42.0 weights [1.058, 0.966, 1.137, 1.044, 1.082, 0.876, 0.838, 1.578, 0.688, 0.711, 1.012, 0.919, 1.09] limit 5.786
num correct 42.0 weights [1.361, 1.053, 1.308, 1.399, 0.139, 1.254, 1.702, 1.593, 1.603, 0.586, 0.568, 0.396, 0.037] limit 6.042
num correct 42.0 weights [0.84, 0.964, 0.526, 1.111, 1.426, 1.096, 1.136, 1.328, 0.929, 0.85, 1.029, 0.489, 1.276] limit 4.969
num correct 42.0 weights [0.912, 0.909, 1.165, 1.137, 1.382, 0.472, 1.562, 1.094, 0.81, 0.521, 1.228, 1.451, 0.358] limit 6.675
num correct 42.0 weights [0.822, 1.167, 1.204, 1.129, 1.159, 0.744, 0.63, 0.857, 0.783, 0.978, 1.208, 1.145, 1.174] limit 5.149
num correct 42.0 weights [0.722, 1.037, 0.781, 1.008, 1.098, 1.111, 1.038, 1.061, 1.019, 1.056, 1.109, 0.961, 0.999] limit 5.564

All 42 predictions were correct in both cases, even when we change the last prediction for 2024. 

Remarks: 

- The weights are normalized to be compatible with the original weights - 1.0 on average.

- The high variation of "fitting" weights shows that Lichtmanns weights and limit value are not special, 
  many other weight sets work equally well - and could be justified for other reasons.  

- If Lichtmanns weights would be "well balanced" you would expect an average weight value of 1.0 if you compute many solutions. 
  But the first key usually gets a much higher weight as the fourth or the sixth one. Interestingly this holds only for the 
  "Harris wins" prediction. 

- If you collect a large number of solutions, the average value for a specific weight could indicate that this average used 
  as weight could improve the reliability of future predictions. 

- Nate Silver already said that "It’s less that he has discovered the right set of keys than that he’s a locksmith and can 
  keep minting new keys until he happens to open all 38 doors". This example shows that you don't need to invent new keys
  to be a "locksmith" opening all doors, including future ones, predicting exactly what you want according to 
  your political agenda. 

-----------------------------------------------------------------------------------------------------------
  
Is there a way we can measure the "robustness" of a model? We could use an epsilon value defining a
limit area [limit-eps, limit+eps] and could maximize this epsilon still maintaining all predictions correct. 

Executing test_harris_prediciton_eps() we get:

Harris looses

num correct 42.0 eps 0.795 weights [2.498, 2.183, 2.07, 0.401, 2.42, 0.032, 0.807, 0.777, 0.136, 1.424, 0.118, 0.087, 0.047] limit 5.254
num correct 42.0 eps 0.806 weights [2.866, 1.82, 2.559, 0.634, 2.667, 0.194, 0.766, 0.706, 0.006, 0.653, 0.025, 0.014, 0.089] limit 5.232
num correct 42.0 eps 0.707 weights [2.128, 1.999, 2.034, 0.467, 1.967, 0.114, 0.928, 0.93, 0.344, 1.366, 0.023, 0.646, 0.054] limit 5.437
num correct 42.0 eps 0.815 weights [2.918, 2.108, 2.256, 0.462, 2.344, 0.152, 0.515, 0.917, 0.109, 0.912, 0.092, 0.057, 0.157] limit 5.328
num correct 42.0 eps 0.806 weights [2.573, 1.622, 2.445, 0.786, 2.038, 0.003, 0.74, 0.741, 0.451, 1.049, 0.345, 0.061, 0.147] limit 5.314
num correct 42.0 eps 0.775 weights [2.351, 2.221, 1.823, 0.219, 2.398, 0.027, 0.838, 0.82, 0.139, 1.508, 0.06, 0.032, 0.564] limit 4.924
num correct 42.0 eps 0.824 weights [2.992, 1.78, 2.444, 0.676, 1.962, 0.038, 0.467, 0.754, 0.151, 0.68, 0.225, 0.241, 0.59] limit 5.515
num correct 42.0 eps 0.855 weights [2.829, 1.766, 2.519, 0.746, 2.268, 0.004, 0.676, 0.754, 0.226, 0.79, 0.113, 0.004, 0.306] limit 5.258
num correct 42.0 eps 0.677 weights [1.961, 1.982, 1.857, 0.421, 2.061, 0.007, 0.849, 0.972, 0.414, 1.429, 0.026, 0.643, 0.377] limit 5.187
num correct 42.0 eps 0.861 weights [2.902, 1.827, 2.402, 0.628, 2.181, 0.039, 0.641, 0.642, 0.597, 0.713, 0.045, 0.144, 0.239] limit 5.289

Harris wins

num correct 42.0 eps 1.366 weights [1.869, 3.041, 0.697, 0.987, 2.705, 0.175, 0.869, 0.909, 0.87, 0.178, 0.003, 0.031, 0.665] limit 5.142
num correct 42.0 eps 1.381 weights [1.779, 2.993, 0.799, 0.865, 2.387, 1.15, 1.03, 0.947, 0.402, 0.018, 0.176, 0.02, 0.434] limit 5.39
num correct 42.0 eps 1.466 weights [1.699, 3.483, 0.404, 0.755, 2.676, 0.727, 1.245, 0.951, 0.854, 0.11, 0.0, 0.036, 0.058] limit 5.4
num correct 42.0 eps 1.311 weights [1.797, 2.869, 0.971, 0.907, 2.094, 0.815, 0.94, 0.936, 0.889, 0.007, 0.007, 0.768, 0.001] limit 5.943
num correct 42.0 eps 1.399 weights [1.941, 3.16, 0.847, 0.873, 2.015, 1.095, 1.025, 1.018, 0.747, 0.002, 0.058, 0.21, 0.009] limit 5.683
num correct 42.0 eps 1.406 weights [1.789, 3.271, 0.473, 0.914, 2.306, 0.671, 1.04, 0.868, 0.947, 0.223, 0.0, 0.014, 0.482] limit 5.17
num correct 42.0 eps 1.407 weights [1.577, 3.531, 0.142, 0.769, 2.159, 0.984, 1.262, 0.806, 1.051, 0.06, 0.148, 0.102, 0.409] limit 5.337
num correct 42.0 eps 1.385 weights [1.547, 3.297, 0.303, 0.747, 3.296, 0.512, 1.249, 0.8, 0.909, 0.154, 0.039, 0.105, 0.045] limit 5.135
num correct 42.0 eps 1.347 weights [2.085, 2.899, 1.094, 0.87, 2.053, 0.834, 1.0, 0.938, 1.11, 0.095, 0.011, 0.007, 0.004] limit 5.439
num correct 42.0 eps 1.349 weights [1.656, 3.134, 0.523, 0.873, 3.05, 0.607, 1.176, 0.876, 0.778, 0.041, 0.013, 0.084, 0.189] limit 5.173
S
till all 42 predictions were correct in both cases, even when we change the last prediction for 2024. 

Remarks: 

- Now it becomes clear that some keys are almost irrelevant.

- Maximizing robustness - epsilon around the limit - leads to quite similar weight values in all optimization runs. 

- The "Harris wins" prediction leads to a much larger epsilon value, so this prediction seems more robust. 
  This result is consistent with Lichtmanns original prediction: Harris wins. 
  In an earlier version of this example because of an error interpreting the prediction was different.
  
---------------------------------------------------------------------------------------------------------------------------------------------

Finally lets maximize the limit area [limit-eps, limit+eps] for all old elections computing the optimal weights 
preserving all correct predictions. Then we compute the weighted sum for the 2024 election. 

num correct 41.0 eps 1.381 weights [1.68, 3.213, 0.461, 0.803, 2.368, 0.804, 1.119, 0.864, 0.771, 0.076, 0.025, 0.731, 0.084] limit 5.822
if sum is > 7.203 challenger wins,  if sum is < 4.441 challenger looses
2024 election, challenger wins = ?, sum = 2.948
num correct 41.0 eps 1.325 weights [2.096, 2.997, 1.06, 0.844, 1.905, 1.221, 0.821, 1.108, 0.782, 0.0, 0.108, 0.052, 0.006] limit 5.575
if sum is > 6.9 challenger wins,  if sum is < 4.249 challenger looses
2024 election, challenger wins = ?, sum = 3.207
num correct 41.0 eps 1.387 weights [1.704, 3.399, 0.47, 0.875, 2.178, 1.329, 1.102, 1.018, 0.516, 0.127, 0.251, 0.016, 0.015] limit 5.551
if sum is > 6.938 challenger wins,  if sum is < 4.163 challenger looses
2024 election, challenger wins = ?, sum = 2.317
num correct 41.0 eps 1.348 weights [2.028, 3.18, 0.65, 0.97, 2.334, 0.73, 0.672, 1.112, 0.322, 0.037, 0.002, 0.01, 0.953] limit 5.19
if sum is > 6.538 challenger wins,  if sum is < 3.843 challenger looses
2024 election, challenger wins = ?, sum = 2.725
num correct 41.0 eps 1.376 weights [1.967, 2.796, 1.004, 1.042, 2.249, 0.663, 0.796, 1.041, 0.671, 0.002, 0.029, 0.0, 0.74] limit 5.213
if sum is > 6.589 challenger wins,  if sum is < 3.838 challenger looses
2024 election, challenger wins = ?, sum = 2.973
num correct 41.0 eps 1.355 weights [1.621, 3.287, 0.604, 0.676, 2.891, 1.056, 1.234, 1.058, 0.205, 0.223, 0.006, 0.089, 0.05] limit 5.369
if sum is > 6.724 challenger wins,  if sum is < 4.013 challenger looses
2024 election, challenger wins = ?, sum = 2.536
num correct 41.0 eps 1.379 weights [1.86, 2.816, 1.065, 1.107, 1.92, 0.985, 1.027, 1.032, 0.986, 0.013, 0.134, 0.048, 0.007] limit 5.557
if sum is > 6.936 challenger wins,  if sum is < 4.178 challenger looses
2024 election, challenger wins = ?, sum = 2.985
num correct 41.0 eps 1.431 weights [1.52, 3.65, 0.015, 0.765, 2.919, 0.782, 1.432, 0.773, 0.885, 0.157, 0.002, 0.001, 0.099] limit 5.166
if sum is > 6.597 challenger wins,  if sum is < 3.735 challenger looses
2024 election, challenger wins = ?, sum = 1.693
num correct 41.0 eps 1.394 weights [1.621, 3.375, 0.396, 0.871, 2.117, 0.833, 1.191, 0.851, 1.274, 0.019, 0.008, 0.01, 0.436] limit 5.197
if sum is > 6.591 challenger wins,  if sum is < 3.803 challenger looses
2024 election, challenger wins = ?, sum = 2.045
num correct 41.0 eps 1.378 weights [1.681, 3.251, 0.512, 0.951, 1.905, 1.064, 1.116, 0.776, 1.227, 0.109, 0.14, 0.016, 0.252] limit 5.319
if sum is > 6.697 challenger wins,  if sum is < 3.941 challenger looses
2024 election, challenger wins = ?, sum = 2.318

So for all 10 runs the weighted sum is way below the threshold limit - eps, so we predict: Harris wins. 
'''

import numpy as np
import sys
from fcmaes import retry
from fcmaes.optimizer import Bite_cpp
from scipy.optimize import Bounds
from functools import partial

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", level="INFO")

def r(v, n=3): return round(v,n)
def rl(l, n=3): return str([r(v, n) for v in l])

def is_false(s):
    if s.lower() == 'true':
        return 0
    elif s.lower() == 'false':
        return 1
    else:
        return None

def read_data(fname):
    
    with open(fname) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            l = line.split()
            if len(l) < 2:
                continue
            if len(l) < 6:
                incumbant = l[-1]
            else:
                winner = l[-1]
                prop_vals = np.array([is_false(s) for s in l if not is_false(s) is None])
                
                #print(prop_vals, sum(prop_vals), incumbant == winner, (incumbant == winner) == (sum(prop_vals) > 5))
                if (incumbant == winner) != (sum(prop_vals) > 5):
                    print("exclude", prop_vals, sum(prop_vals), incumbant == winner, (incumbant == winner) == (sum(prop_vals) > 5), file=sys.stderr)
                else: # use only correct predictions:
                    data.append((prop_vals, incumbant == winner))                
        return data

def collect_all_data():
    data_old = read_data('data_old.txt')
    data_new = read_data('data_new.txt')
    data = data_old + data_new
    # check all predictions
    for props, inc_is_winner in data:
        if inc_is_winner != (sum(props) > 5):
            print("error, sum props", sum(props), "incumbant is winner", inc_is_winner, file=sys.stderr)
    return data

# optimize the correct predictions for a given model defined by its weights + limit
def fit(n, props, inc_is_winner, x):
    weights = x[:-1]
    limit = x[-1]
    weights *= len(weights)/sum(weights) # normalize weights so that sum is num_props
    weighted_props = weights*props
    num_correct = sum( [ inc_is_winner[i] == (sum(weighted_props[i]) > limit) for i in range(n)] )
    return -num_correct

# show 10 solution models for "Harris wins" and 10 for "Harris looses"
def find_other_model(data, harris_predict = True):
    
    props = np.array([d[0] for d in data])
    inc_is_winner = np.array([d[1] for d in data])
    # set for harris - last prediction
    inc_is_winner[-1] = harris_predict
    
    n = len(data)
    num_props = 13
    dim = num_props + 1
    lb = [0.0]*dim
    ub = [1.0]*num_props + [13.0]
    bounds = Bounds(lb,ub)
       
    for i in range(10):
        ret = retry.minimize(partial(fit, n, props, inc_is_winner), bounds, optimizer=Bite_cpp(1000),
                   num_retries=32)
        weights = ret.x[:-1]
        limit = ret.x[-1]
        weights *= num_props/sum(weights) # normalize weights so that sum is num_props
        print ("num correct", -ret.fun, "weights", rl(list(weights)), "limit", r(limit))

def test_harris_prediciton():
    data = collect_all_data()
    print("Harris looses")
    find_other_model(data, harris_predict = True) # challenger (Trump) wins
    print("Harris wins")
    find_other_model(data, harris_predict = False) # challenger (Trump) looses

# optimize the correct predictions for a given model defined by its weights + limit
# maximize an epsilon around the limit to improve robustness
def fit_eps(n, props, inc_is_winner, x):
    weights = x[:-2]
    eps = x[-1]
    limit = x[-2]
    weights *= len(weights)/sum(weights) # normalize weights so that sum is num_props
    weighted_props = weights*props
    num_correct1 = sum( [ inc_is_winner[i] == (sum(weighted_props[i]) > limit + eps) for i in range(n)] )
    num_correct2 = sum( [ (not inc_is_winner[i]) == (sum(weighted_props[i]) < limit - eps) for i in range(n)] )
    return - (num_correct1 + num_correct2 + 0.1*eps)

# show 10 solution models for "Harris wins" and 10 for "Harris looses" thereby maximizing epsilon
def find_other_model_eps(data, harris_predict = True):
    
    props = np.array([d[0] for d in data])
    inc_is_winner = np.array([d[1] for d in data])
    # set for harris - last prediction
    inc_is_winner[-1] = harris_predict
    
    n = len(data)
    num_props = 13
    dim = num_props + 2
    lb = [0.0]*dim
    ub = [1.0]*num_props + [13.0] + [4.999]
    bounds = Bounds(lb,ub)
       
    for i in range(10):
        ret = retry.minimize(partial(fit_eps, n, props, inc_is_winner), bounds, optimizer=Bite_cpp(5000),
                   num_retries=32)
        weights = ret.x[:-2]
        limit = ret.x[-2]
        eps = ret.x[-1]
        weights *= num_props/sum(weights) # normalize weights so that sum is num_props
        corr = int(-ret.fun) / 2
        print ("num correct", corr, "eps", r(eps), "weights", rl(list(weights)), "limit", r(limit))

def test_harris_prediciton_eps():
    data = collect_all_data()
    print("Harris looses")
    find_other_model_eps(data, harris_predict = True) # challenger (Trump) wins
    print("Harris wins")
    find_other_model_eps(data, harris_predict = False) # challenger (Trump) looses

def single_sum(n, single_props, weights):
    weights *= len(weights)/sum(weights) # normalize weights so that sum is num_props
    weighted_props = weights*single_props
    return np.sum(weighted_props)

def find_harris_estimate_eps(data):
    
    harris_data = data[-1:] # harris props
    single_props = harris_data[0][0]
    
    data = data[:-1] # remove harris result
    props = np.array([d[0] for d in data])
    inc_is_winner = np.array([d[1] for d in data])
    
    n = len(data)
    num_props = 13
    dim = num_props + 2
    lb = [0.0]*dim
    ub = [1.0]*num_props + [13.0] + [4.999]
    bounds = Bounds(lb,ub)
       
    for i in range(10):
        ret = retry.minimize(partial(fit_eps, n, props, inc_is_winner), bounds, optimizer=Bite_cpp(5000),
                   num_retries=32)
        weights = ret.x[:-2]
        limit = ret.x[-2]
        eps = ret.x[-1]
        weights *= num_props/sum(weights) # normalize weights so that sum is num_props
        corr = int(-ret.fun) / 2
        print ("num correct", corr, "eps", r(eps), "weights", rl(list(weights)), "limit", r(limit))
        print (f'if sum is > {r(limit+eps)} challenger wins,  if sum is < {r(limit-eps)} challenger looses')
        # for i in range(len(data)): # uncomment to verify old predictions
        #     print(f'election {i}, challenger wins = {inc_is_winner[i]}, sum = {r(single_sum(n, props[i], weights))}')
        print(f'2024 election, challenger wins = ?, sum = {r(single_sum(n, single_props, weights))}')

def check(n, props, inc_is_winner, weights, eps, limit):
    weights *= len(props)/sum(weights) # normalize weights so that sum is num_props
    weighted_props = weights*props
    num_correct1 = sum( [ inc_is_winner[i] == (sum(weighted_props[i]) > limit + eps) for i in range(n)] )
    num_correct2 = sum( [ (not inc_is_winner[i]) == (sum(weighted_props[i]) < limit - eps) for i in range(n)] )
    return - (num_correct1 + num_correct2 + 0.1*eps)


def test_harris_estimate_eps():
    data = collect_all_data()
    print("without Harris election")
    find_harris_estimate_eps(data) 

if __name__ == '__main__':
    test_harris_prediciton()
    test_harris_prediciton_eps()
    test_harris_estimate_eps()
    pass