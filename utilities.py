import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np

def simple_mlp(input_tensor, n_in, n_hid, n_out, scope):
    # hidden layer
    with tf.name_scope(scope):
        with tf.name_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal([n_in, n_hid],
                                    stddev=1.0 / math.sqrt(float(n_in))),
                name='weights')
            biases = tf.Variable(tf.zeros([n_hid]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        # output layer
        with tf.name_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([n_hid, n_out],
                                    stddev=1.0 / math.sqrt(float(n_hid))),
                name='weights')
            biases = tf.Variable(tf.zeros([n_out]),
                                 name='biases')
            out_tensor = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    return out_tensor


def plot_img_mats(mat):
    # plot l*m*n mats as l m by n gray-scale images
    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n // cols))

    plt.style.use('grayscale')
    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()

    for idx, ax in enumerate(ax_list):
        if idx >= n:
            ax.axis('off')
        else:
            ax.imshow(mat[idx, :, :], interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

