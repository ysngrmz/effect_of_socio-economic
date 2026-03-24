import sys

import numpy as np

sys.path.append('/truba/home/kokumusdagdeler/yasin/effect_of_socio/scripts/')
import read_data as rd
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmet
import KTBoost.KTBoost as KTBoost


from interpret import show

import pandas as pd

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import pickle


learning_rate_values = Real(low=0.001, high=0.3, name="learning_rate")
max_depth_values = Integer(low=2, high=10, name="max_depth")
min_samples_leaf_values = Integer(low=1, high=10, name="min_samples_leaf")

param_grid = [learning_rate_values, max_depth_values, min_samples_leaf_values]

X_train = []
y_train = []
X_test = []
y_test = []
X_val = []
y_val = []

best_mae = float('inf')
best_model = KTBoost.BoostingRegressor()
best_test_pred = []
best_param_list = []


@use_named_args(dimensions=param_grid)
def call_model(learning_rate, max_depth, min_samples_leaf):
    global X_train, y_train, X_test, y_test, X_val, y_val
    global best_model, best_mae, best_test_pred, best_param_list

    regr = KTBoost.BoostingRegressor(learning_rate=learning_rate,
                             max_depth = max_depth,
                             min_samples_leaf=min_samples_leaf
    )

    regr.fit(X_train, y_train)
    val_predict = regr.predict(X_val)
    test_predict = regr.predict(X_test)
    val_mae = skmet.mean_absolute_error(y_val, val_predict)

    if val_mae < best_mae:
        best_mae = val_mae
        best_model = regr
        best_test_pred = test_predict
        best_param_list = [["learning_rate", learning_rate], 
                           ["max_depth", max_depth], 
                           ["min_samples_leaf",min_samples_leaf]]

    return val_mae


inputs, targets, columns, col_val_list = rd.read_data(data_read_dir='/truba/home/kokumusdagdeler/yasin/effect_of_socio/data/')

inputs = pd.DataFrame(inputs, columns=columns)

n_of_holdout = 200

for holdout in range(1, n_of_holdout+1):
    best_mae = float('inf')
    best_model = KTBoost.BoostingRegressor()
    best_test_pred = []
    best_param_list = []

    X_train, X_test, y_train, y_test = train_test_split(inputs,
                                                        targets,
                                                        test_size=0.1)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.11)

    sr = gp_minimize(call_model, dimensions=param_grid, acq_func='EI', n_calls=50)

    test_mae = skmet.mean_absolute_error(y_test, best_test_pred)

    np.savetxt("holdout_results/targets_for_holdout_" + str(holdout), y_test)
    np.savetxt("holdout_results/preds_for_holdout_" + str(holdout), best_test_pred)
    np.savetxt("holdout_results/best_params_for_holdout_" + str(holdout), best_param_list, fmt="%s")
    np.savetxt("holdout_results/X_train_for_holdout_" + str(holdout), X_train)
    pickle.dump(best_model, open("holdout_results/best_model_for_holdout_" + str(holdout), 'wb'))

    print(100 * "*")
    print("TEST MAE For Holdout", holdout, ":", test_mae)
    print(100 * "*")

