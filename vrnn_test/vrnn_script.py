from __future__ import print_function
from vrnn_train import run_training, run_generation, run_read_then_continue
from number_series_gen import *
from params import PARAM_DICT
import numpy as np

mode = 1

if mode == 0:  # create a number series according to specifications
    number = PARAM_DICT['num_batches'] * PARAM_DICT['batch_size']
    length = PARAM_DICT['seq_length']
    dim = PARAM_DICT['data_dim']
    file_path = PARAM_DICT['data_path']
    sid = PARAM_DICT['series']
    save_series(number=number, length=length, dim=dim, file_path=file_path, sid=sid)

elif mode == 1:  # run training
    run_training(PARAM_DICT)

elif mode == 2:  # run generation, then series check on results
    x = run_generation(PARAM_DICT['log_path'] + '/params.pkl')
    sid = PARAM_DICT['series']
    for idx in range(10):
        series_check(x[:, idx, :], sid)
elif mode == 3:  # run read then continue, then series check on results
    data = np.load(PARAM_DICT['data_path'])
    read_seq = data[:, 0, :]  # first series
    x = run_read_then_continue(PARAM_DICT['log_path'] + '/params.pkl', read_seq, ckpt_file=None, batch_size=1)
    sid = PARAM_DICT['series']
    for idx in range(10):
        series_check(x[:, idx, :], sid)
else:  # run series check on data
    data = np.load(PARAM_DICT['data_path'])
    for idx in range(5):
        series_check(data[:, idx, :], PARAM_DICT['series'])
