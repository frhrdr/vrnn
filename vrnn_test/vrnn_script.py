from __future__ import print_function
from vrnn_train import run_training, run_generation
from number_series_gen import *
from params import PARAM_DICT
import numpy as np

# save_series3(1000, 10, 10, 'data/series4_1000n_10d_10t.npy')

# run_training(PARAM_DICT)

x = run_generation('data/logs/test1/params.pkl')
# # print(x)
# print(x.shape)
# acc = np.zeros((6,))
for idx in range(5):  # x.shape[1]):
    series4_check(x[:, idx, :])
    # print(series2_check(x[:, idx, :]))
#     acc += series2_check(x[:, idx, :])
# print(acc / float(x.shape[1]))

# data = np.load('data/series2_5000n_10d_10t.npy')
# acc = np.zeros((6,))
# for idx in range(data.shape[1]):
#     # print(series2_check(data[:, idx, :]))
#     acc += series2_check(data[:, idx, :])
# print(acc / float(data.shape[1]))
