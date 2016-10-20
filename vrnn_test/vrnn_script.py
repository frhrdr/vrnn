from __future__ import print_function
from vrnn_train import run_training, run_generation
from number_series_gen import *
from params import PARAM_DICT
import numpy as np

# save_series(2000, 10, 10, 'data/series5_2000n_10t_10d.npy', 5)

# run_training(PARAM_DICT)

# x = run_generation('data/logs/test1/params.pkl')
# # print(x)
# print(x.shape)
# acc = np.zeros((6,))
# for idx in range(5):  # x.shape[1]):
#     series5_check(x[:, idx, :])

data = np.load('data/series5_2000n_10t_10d.npy')
for idx in range(data.shape[1]):
    series5_check(data[:, idx, :])
