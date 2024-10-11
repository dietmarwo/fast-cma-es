
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

Executing the code below we get:

Harris wins

num correct 42.0 weights [1.842, 2.288, 1.769, 0.109, 1.485, 0.366, 1.039, 0.815, 0.025, 1.794, 0.221, 0.396, 0.85] limit 5.024
num correct 42.0 weights [1.42, 1.656, 0.952, 0.329, 1.454, 0.075, 1.085, 0.722, 1.833, 1.835, 0.124, 0.518, 0.996] limit 4.62
num correct 42.0 weights [1.436, 0.748, 1.913, 0.836, 2.06, 0.147, 0.801, 0.987, 0.798, 1.392, 0.403, 0.788, 0.691] limit 4.773
num correct 42.0 weights [1.333, 0.97, 1.31, 0.918, 1.655, 0.219, 1.084, 1.155, 0.462, 1.19, 0.016, 1.598, 1.09] limit 5.292
num correct 42.0 weights [1.958, 1.419, 1.896, 0.512, 1.021, 0.247, 1.146, 0.767, 0.099, 0.885, 0.063, 1.887, 1.1] limit 5.85
num correct 42.0 weights [1.606, 1.541, 1.587, 0.75, 0.999, 0.066, 0.796, 0.706, 1.021, 1.185, 0.013, 1.174, 1.555] limit 5.237
num correct 42.0 weights [1.837, 1.633, 1.203, 0.174, 1.178, 0.174, 1.139, 0.742, 1.131, 1.145, 0.363, 1.965, 0.316] limit 6.146
num correct 42.0 weights [1.563, 1.201, 1.54, 0.731, 1.412, 0.426, 0.461, 0.935, 1.396, 0.89, 0.478, 1.609, 0.359] limit 5.493
num correct 42.0 weights [1.825, 1.336, 1.858, 0.638, 1.601, 0.188, 0.757, 0.321, 0.818, 1.064, 0.287, 0.711, 1.595] limit 4.924
num correct 42.0 weights [1.588, 1.644, 1.692, 0.47, 1.355, 0.068, 0.836, 1.268, 0.515, 1.699, 0.507, 0.496, 0.863] limit 4.777

Harris looses

num correct 42.0 weights [1.338, 1.738, 0.572, 0.846, 1.953, 0.622, 1.388, 1.346, 0.478, 1.334, 0.395, 0.681, 0.31] limit 5.366
num correct 42.0 weights [1.06, 1.359, 0.975, 1.32, 1.43, 1.288, 1.253, 0.703, 0.743, 0.957, 0.451, 1.08, 0.381] limit 5.789
num correct 42.0 weights [0.65, 1.443, 0.835, 1.192, 1.005, 0.881, 0.805, 1.003, 0.588, 1.046, 0.753, 0.903, 1.898] limit 5.365
num correct 42.0 weights [1.189, 1.386, 0.558, 0.494, 1.129, 1.529, 1.428, 1.013, 0.8, 1.0, 1.148, 0.07, 1.257] limit 5.276
num correct 42.0 weights [1.316, 1.243, 0.691, 1.206, 1.532, 0.749, 1.463, 0.918, 0.862, 1.131, 1.169, 0.121, 0.599] limit 5.422
num correct 42.0 weights [0.987, 1.246, 1.071, 1.168, 0.686, 0.962, 1.315, 1.178, 0.558, 1.246, 0.747, 0.734, 1.101] limit 5.582
num correct 42.0 weights [1.444, 0.88, 1.359, 1.296, 0.798, 0.965, 0.988, 1.143, 0.743, 0.697, 0.941, 0.678, 1.068] limit 5.446
num correct 42.0 weights [0.968, 1.157, 0.875, 0.609, 1.102, 1.081, 1.034, 0.967, 0.828, 1.162, 1.14, 1.13, 0.948] limit 5.338
num correct 42.0 weights [1.202, 0.71, 0.996, 0.951, 0.913, 1.239, 0.878, 0.772, 1.142, 0.967, 1.097, 1.064, 1.07] limit 5.693
num correct 42.0 weights [0.97, 1.55, 0.544, 0.824, 0.553, 1.859, 1.514, 0.97, 0.673, 0.082, 1.574, 1.34, 0.547] limit 6.879

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

def fit(n, props, inc_is_winner, x):
    weights = x[:-1]
    limit = x[-1]
    weights *= len(props)/sum(weights) # normalize weights so that sum is num_props
    weighted_props = weights*props
    num_correct = sum( [ inc_is_winner[i] == (sum(weighted_props[i]) > limit) for i in range(n)] )
    return -num_correct

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

if __name__ == '__main__':
    data = collect_all_data()
    print("Harris wins")
    find_other_model(data, harris_predict = True) # Harris wins
    print("Harris looses")
    find_other_model(data, harris_predict = False) # Harris looses
    pass