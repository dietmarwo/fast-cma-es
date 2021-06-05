# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# adapted from https://www.kaggle.com/pablocastilla/predict-house-prices-with-xgboost-regression

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV  #Performing grid search
from collections import OrderedDict

from xgboost import XGBRegressor
from fcmaes.optimizer import dtime
import multiprocessing as mp
import ctypes as ct
import math, time

# download data from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train_dataset=pd.read_csv('../input/train.csv', header=0)
test_dataset=pd.read_csv('../input/test.csv', header=0)

categorical_features=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
                     'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
                     'MiscFeature','SaleType','SaleCondition']
every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]
train_dataset.describe()

every_column_non_categorical= [col for col in train_dataset.columns if col not in categorical_features and col not in ['Id'] ]

numeric_feats = train_dataset[every_column_non_categorical].dtypes[train_dataset.dtypes != "object"].index

train_dataset[numeric_feats] = np.log1p(train_dataset[numeric_feats])

every_column_non_categorical= [col for col in test_dataset.columns if col not in categorical_features and col not in ['Id'] ]
numeric_feats = test_dataset[every_column_non_categorical].dtypes[test_dataset.dtypes != "object"].index
test_dataset[numeric_feats] = np.log1p(test_dataset[numeric_feats])

features_with_nan=['Alley','MasVnrType','BsmtQual','BsmtQual','BsmtCond','BsmtCond','BsmtExposure',
                   'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']
#function that creates a column for every value it might have
def ConverNaNToNAString(data, columnList):
    for x in columnList:       
        data[x] =str(data[x])              
            
ConverNaNToNAString(train_dataset, features_with_nan)
ConverNaNToNAString(test_dataset, features_with_nan)

train_dataset = pd.get_dummies(train_dataset,columns = categorical_features)
test_dataset = pd.get_dummies(test_dataset,columns = categorical_features)

model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 

every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]
model.fit(train_dataset[every_column_except_y],train_dataset['SalePrice'])

weighted_features = model.get_booster().get_score(importance_type='weight').items()

ordered_features = OrderedDict(sorted(weighted_features, key=lambda t: t[1], reverse=True))

most_relevant_features= list( dict((k, v) for k, v in weighted_features if v >= 10).keys())

#removing outliers
train_dataset = train_dataset[train_dataset.GrLivArea < 8.25]
train_dataset = train_dataset[train_dataset.LotArea < 11.5]
train_dataset = train_dataset[train_dataset.SalePrice < 13]
train_dataset = train_dataset[train_dataset.SalePrice > 10.75]
train_dataset.drop("Id", axis=1, inplace=True)

train_x=train_dataset[most_relevant_features]
train_y=train_dataset['SalePrice']

grid_search = False
hyperopt = False
bayesian = False
evolutionary = True

# grid search

if grid_search:
    parameters_for_testing = {
        'colsample_bytree':[0.4,0.6,0.8],
        'gamma':[0,0.03,0.1,0.3],
        'min_child_weight':[1.5,6,10],
        'learning_rate':[0.1,0.07],
        'max_depth':[3,5],
        'n_estimators':[10000],
        'reg_alpha':[1e-5, 1e-2,  0.75],
        'reg_lambda':[1e-5, 1e-2, 0.45],
        'subsample':[0.6,0.95]  
    }

    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)
 
    gsearch = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, 
                            n_jobs=6,iid=False, verbose=100, scoring='neg_mean_squared_error')
    gsearch.fit(train_x, train_y)

    #visualize the best couple of parameters
    print (gsearch.best_params_) 

from fcmaes.optimizer import logger
logger = logger("house.log")

# Optimization objective 

def cv_score(X):
    X = X[0]     
    score = cross_val_score(
            XGBRegressor(colsample_bytree=X[0],
                         gamma=X[1],                 
                         min_child_weight=X[2],
                         learning_rate=X[3],
                         max_depth=int(X[4]),
                         n_estimators=10000,                                                                    
                         reg_alpha=X[5],
                         reg_lambda=X[6],
                         subsample=X[7], 
                         n_jobs=1 # required for cmaes with multiple workers
                         ), 
                train_x, train_y, scoring='neg_mean_squared_error').mean()

    return np.array(score)

def obj_f(X):
    return -cv_score([X])

class cv_problem(object):

    def __init__(self, pfun, bounds):
        self.name = "cv_score"
        self.dim = len(bounds.lb)
        self.pfun = pfun
        self.bounds = bounds
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.t0 = time.perf_counter()

    def fun(self, x):
        self.evals.value += 1
        y = self.pfun(x)
        if y < self.best_y.value:
            self.best_y.value = y
            logger.info(str(dtime(self.t0)) + ' '  + 
                          str(self.evals.value) + ' ' + 
                          str(self.best_y.value) + ' ' + 
                          str(list(x)))
        return y

# hyperopt TPE

if hyperopt:

    from hyperopt import fmin, hp, tpe, STATUS_OK
    
    def obj_fmin(X):
        return {'loss': -np.asscalar(cv_score([X])), 'status': STATUS_OK }
     
    xgb_space = [
            hp.quniform('colsample_bytree', 0.4, 0.8, 0.05),
            hp.quniform('gamma', 0, 0.3, 0.05),
            hp.quniform('min_child_weight', 1.5, 10, 0.5),
            hp.quniform('learning_rate', 0.07, 0.1, 0.05),
            hp.choice('max_depth', [3,4,5]),
            hp.uniform('reg_alpha', 1e-5, 0.75),
            hp.uniform('reg_lambda', 1e-5, 0.45),
            hp.uniform('subsample', 0.6, 0.95)]
     
    best = fmin(fn = obj_fmin, space = xgb_space, algo = tpe.suggest, 
                    max_evals = 2000, verbose=False)


# Bayesian optimization

if bayesian:

    from GPyOpt.methods import BayesianOptimization

    bds = [ {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.4, 0.8)},
            {'name': 'gamma', 'type': 'continuous', 'domain': (0, 0.3)},
            {'name': 'min_child_weight', 'type': 'continuous', 'domain': (1.5, 10)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.07, 0.1)},
            {'name': 'max_depth', 'type': 'discrete', 'domain': (3, 5)},
            {'name': 'reg_alpha', 'type': 'continuous', 'domain': (1e-5, 0.75)},
            {'name': 'reg_lambda', 'type': 'continuous', 'domain': (1e-5, 0.45)},
            {'name': 'subsample', 'type': 'continuous', 'domain': (0.6, 0.95)}]
    
    optimizer = BayesianOptimization(f=cv_score, 
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.05,
                                     exact_feval=True, 
                                     maximize=True)
     
    optimizer.run_optimization(max_iter=20000)
    y_bo = np.maximum.accumulate(-optimizer.Y).ravel()
    print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

# standard evolutionary algorithms

if evolutionary and __name__ == '__main__':
    mp.freeze_support()

    from scipy.optimize import Bounds
    from fcmaes import decpp, cmaescpp, bitecpp, de, cmaes
    from fcmaes import cmaes 
    from fcmaes import de
    
    bounds = Bounds([0.4, 0, 1.5, 0.07, 3, 1e-5, 1e-5, 0.6], [0.8, 0.3, 10, 0.1, 5.99, 0.75, 0.45, 0.95])
 
    problem = cv_problem(obj_f, bounds)
    #ret = bitecpp.minimize(problem.fun, problem.bounds, max_evaluations = 20000)
    
    # for cmaescpp, cmaes and de with multiple workers set n_jobs=1 in XGBRegressor
    
    #ret = cmaescpp.minimize(problem.fun, problem.bounds, popsize=32, max_evaluations = 20000, workers=mp.cpu_count())
    #ret = decpp.minimize(problem.fun, problem.dim, problem.bounds, popsize=16, max_evaluations = 20000)
    
    # delayed state update
    ret = cmaes.minimize(problem.fun, problem.bounds, popsize=16, max_evaluations = 20000, 
                          workers=mp.cpu_count(), delayed_update=True)
    
    #ret = de.minimize(problem.fun, problem.dim, problem.bounds, popsize = 16, max_evaluations = 20000, workers=mp.cpu_count())


    

