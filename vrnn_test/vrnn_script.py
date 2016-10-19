from __future__ import print_function
from vrnn_train import run_training, run_generation
from number_series_gen import series2_check, save_series2
from params import PARAM_DICT


# save_series2(1000, 20, 5, 'data/series2_1000n_20d_5t.npy')

run_training(PARAM_DICT)

# x = run_generation('data/logs/test1/params.pkl')
# # print(x)
# print(x.shape)
# for idx in range(x.shape[1]):
#     print(series2_check(x[:, idx, :]))
