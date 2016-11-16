from __future__ import print_function
from vrnn_train import run_training, run_generation
from number_series_gen import *
from params import PARAM_DICT
import numpy as np

mode = 2

if mode == 0:
    number = PARAM_DICT['num_batches'] * PARAM_DICT['batch_size']
    length = PARAM_DICT['seq_length']
    dim = PARAM_DICT['data_dim']
    file_path = PARAM_DICT['data_path']
    sid = PARAM_DICT['series']
    save_series(number=number, length=length, dim=dim, file_path=file_path, sid=sid)
elif mode == 1:
    run_training(PARAM_DICT)
elif mode == 2:
    x = run_generation(PARAM_DICT['log_path'] + '/params.pkl')
    sid = PARAM_DICT['series']
    for idx in range(10):
        series_check(x[:, idx, :], sid)
else:
    data = np.load(PARAM_DICT['data_path'])
    for idx in range(5):
        series_check(data[:, idx, :], PARAM_DICT['series'])
