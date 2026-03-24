import sys

import numpy as np

sys.path.append('/truba/home/kokumusdagdeler/yasin/effect_of_socio/scripts/')
import read_data as rd
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmet
from sklearn import linear_model


from interpret import show

import pandas as pd

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import pickle


max_iter_values = Integer(low=100, high=1000, name="max_iter")
tol_values = Real(low=1e-5, high=1e-1, name="tol")
alpha_1_values = Real(low=1e-10, high=1e-3, name="alpha_1")
alpha_2_values = Real(low=1e-10, high=1e-3, name="alpha_2")
lambda_1_values = Real(low=1e-10, high=1e-3, name="lambda_1")
lambda_2_values = Real(low=1e-10, high=1e-3, name="lambda_2")

param_grid = [max_iter_values, tol_values, alpha_1_values, alpha_2_values, lambda_1_values, lambda_2_values]

X_train = []
y_train = []
X_test = []
y_test = []
X_val = []
y_val = []

best_mae = float('inf')
best_model = linear_model.BayesianRidge()
best_test_pred = []
best_param_list = []


@use_named_args(dimensions=param_grid)
def call_model(max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2):
    global X_train, y_train, X_test, y_test, X_val, y_val
    global best_model, best_mae, best_test_pred, best_param_list

    regr = linear_model.BayesianRidge(max_iter=max_iter,
                             tol=tol,
                             alpha_1=alpha_1,
                             alpha_2=alpha_2,
                             lambda_1=lambda_1,
                             lambda_2=lambda_2
    )

    regr.fit(X_train, y_train)
    val_predict = regr.predict(X_val)
    test_predict = regr.predict(X_test)
    val_mae = skmet.mean_absolute_error(y_val, val_predict)

    if val_mae < best_mae:
        best_mae = val_mae
        best_model = regr
        best_test_pred = test_predict
        best_param_list = [["max_iter", max_iter], 
                           ["tol",tol],
                           ["alpha_1",alpha_1],
                           ["alpha_2",alpha_2],
                           ["lambda_1",lambda_1],
                           ["lambda_2",lambda_2]]

    return val_mae


inputs, targets, columns, col_val_list = rd.read_data(data_read_dir='/truba/home/kokumusdagdeler/yasin/effect_of_socio/data/')

inputs = pd.DataFrame(inputs, columns=columns)

n_of_holdout = 200

for holdout in range(1, n_of_holdout+1):
    best_mae = float('inf')
    best_model = linear_model.BayesianRidge()
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

