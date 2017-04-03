from __future__ import print_function
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# from utilities import plot_img_mats

# batch_size = 16
# data = input_data.read_data_sets('data/mnist/').train
# x = np.reshape(data.next_batch(batch_size)[0], (batch_size, 28, 28))
# x = np.transpose(x, (1, 0, 2))
# x = np.transpose(x, (1, 0, 2))
# plot_img_mats(x[:16, :, :])

a = np.load('data/logs/mnist_10/params.pkl')
for key in a:
    print(key, ': ', a[key])

