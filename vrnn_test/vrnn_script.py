from __future__ import print_function
from vrnn_train import run_training, run_generation
from number_series_gen import *
from params import PARAM_DICT
from iamondb_reader import mat_to_plot
import numpy as np

mode = 1

if mode == 1:  # run training
    run_training(PARAM_DICT)
elif mode == 2:  # run generation, then plot the results
    x = run_generation(PARAM_DICT['log_path'] + '/params.pkl', ckpt_file=PARAM_DICT['log_path'] + '/ckpt-3000')

    # mask 200 cut
    # [ 7.65791469  0.54339499  0.03887757]
    # [ 33.82594281  36.81890347   0.19330315]
    # meanx = 7.65791469
    # meany = 0.54339499
    # stdx = 33.82594281
    # stdy = 36.81890347

    # no mask 200 cut
    # [ 7.61830955  0.54058467  0.03867651]
    # [ 33.74283029  36.72359088   0.19282281]
    meanx = 7.61830955
    meany = 0.54058467
    stdx = 33.74283029
    stdy = 36.72359088

    print(x.shape)
    for idx in range(10):
        mat_to_plot(x[:, idx, :], meanx=meanx, meany=meany, stdx=stdx, stdy=stdy)


# if mode == 0:  # create a number series according to specifications
#     number = PARAM_DICT['num_batches'] * PARAM_DICT['batch_size']
#     length = PARAM_DICT['seq_length']
#     dim = PARAM_DICT['data_dim']
#     file_path = PARAM_DICT['data_path']
#     sid = PARAM_DICT['series']
#     save_series(number=number, length=length, dim=dim, file_path=file_path, sid=sid)
# elif mode == 2:  # run generation, then series check on results
#     x = run_generation(PARAM_DICT['log_path'] + '/params.pkl')
#     sid = PARAM_DICT['series']
#     for idx in range(10):
#         series_check(x[:, idx, :], sid)
# elif mode == 3:  # run series check on data
#     data = np.load(PARAM_DICT['data_path'])
#     for idx in range(5):
#         series_check(data[:, idx, :], PARAM_DICT['series'])