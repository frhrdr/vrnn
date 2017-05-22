from vrnn_train import run_training, run_generation, run_read_then_continue
from params import PARAM_DICT
from iamondb_reader import mat_to_plot
from utilities import plot_img_mats
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mode = 1

if mode == 1:  # run training
    run_training(PARAM_DICT)
elif mode == 2:  # run mnist generation
    x = run_generation(PARAM_DICT['log_path'] + 'params.pkl', ckpt_file=PARAM_DICT['log_path'] + 'ckpt-20000')
    x = np.transpose(x, (1, 0, 2))
    plot_img_mats(x[:36, :, :])
elif mode == 3:  # run mnist generation with prime sequence
    batch_size = 36
    cut_after = 7
    data = input_data.read_data_sets('data/mnist/').validation
    x = np.reshape(data.next_batch(batch_size)[0], (batch_size, 28, 28))
    x = np.transpose(x, (1, 0, 2))
    x = x[:cut_after, :, :]
    y = run_read_then_continue(PARAM_DICT['log_path'] + 'params.pkl',
                               ckpt_file=PARAM_DICT['log_path'] + 'ckpt-20000',
                               read_seq=x, batch_size=batch_size)
    z = np.concatenate([x, y], axis=0)
    z = np.transpose(z, (1, 0, 2))
    plot_img_mats(z)

elif mode == 4:  # run handwriting generation, then plot the results
    x = run_generation(PARAM_DICT['log_path'] + 'params.pkl', ckpt_file=PARAM_DICT['log_path'] + 'ckpt-9500')

    # mask 500 cut mean and variance
    m = [7.97040614, 0.29582727]
    s = [34.80169994, 36.07062753]

    for idx in range(5):
        # print(x[:, idx, :])
        mat_to_plot(x[:, idx, :], meanx=m[0], meany=m[1], stdx=s[0], stdy=s[1])

