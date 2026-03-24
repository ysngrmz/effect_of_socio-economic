import pandas as pd
import numpy as np


def read_data(data_read_dir="../data/",
              data_read_file="ds_v1.xlsx"):

    full_data = pd.read_excel(data_read_dir + data_read_file)

    col_names = full_data.columns.tolist()[1:]

    full_data_values = full_data.values[:, 1:]
    string_inds = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]
    col_val_list = []

    for string_ind in string_inds:
        current_col = full_data_values[:, string_ind]
        current_col = current_col.astype(str)
        unique_col_names, int_representation = np.unique(current_col, return_inverse=True)
        col_val_list.append(unique_col_names.tolist())
        full_data_values[:, string_ind] = int_representation
    for ind in range(1, full_data_values.shape[1]):
        full_data_values[:, ind] = min_max_scale(full_data_values[:, ind])

    inputs = full_data_values[:, 1:]
    targets = full_data_values[:, 0]

    col_names = col_names[1:]
    for i in range(len(col_names)):
        col_names[i] = col_names[i].replace("\n", " ")

    return inputs, targets, col_names, col_val_list


def min_max_scale(X, range=(0, 1)):
    mi, ma = range
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled
