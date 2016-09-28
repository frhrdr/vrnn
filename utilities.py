import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class NetGen:

    def __init__(self):
        self.fd = {}                    # function dict stores generator functions under name
        self.td = defaultdict(list)     # tensor dict stores list of associated varable tensors under function name

    def __str__(self):
        return str(self.fd.keys())

    # adds net-gen function of certain type to the internal dict
    def add_net(self, name, nn_type, params):

        if nn_type is 'simple_mlp':
            n_in, n_hid, n_out = params['layers']

            def f(in_pl):
                return simple_mlp(in_pl, n_in, n_hid, n_out, name, self.td[name])
            self.fd[name] = f

    # concatenates several tensors into one input to existing nn of given name
    def weave_inputs(self, name, in_dims):
        f = self.fd[name]

        def g(*args):
            # TODO: maybe make dimension flexible
            in_pl = tf.concat(1, list(args), name=name + "_joint_inputs")
            return f(in_pl)

        self.fd[name] = g


def simple_mlp(input_tensor, n_in, n_hid, n_out, scope, var_list):

    idx = running_idx()
    store_vars = len(var_list) == 0

    # hidden layer
    with tf.name_scope(scope):
        with tf.name_scope('hidden'):
            if store_vars:
                var_list.add(tf.Variable(tf.truncated_normal([n_in, n_hid],
                                                             stddev=1.0 / math.sqrt(float(n_in))),
                                         name='weights'))

            weights = var_list[idx.next()]

            if store_vars:
                var_list.add(tf.Variable(tf.zeros([n_hid]), name='biases'))
            biases = var_list[idx.next()]

            hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        # output layer
        with tf.name_scope('output'):
            if store_vars:
                var_list.add(tf.Variable(tf.truncated_normal([n_hid, n_out],
                                                             stddev=1.0 / math.sqrt(float(n_in))),
                                         name='weights'))
            weights = var_list[idx.next()]

            if store_vars:
                var_list.add(tf.Variable(tf.zeros([n_out]), name='biases'))
            biases = var_list[idx.next()]
            out_tensor = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    return out_tensor


def general_mlp(input_tensor, layers, name):

    n_hidden_layers = len(layers) - 2
    last = input_tensor
    with tf.name_scope(name):
        # stack hidden layers
        for idx in range(n_hidden_layers):
            with tf.name_scope('hidden' + str(idx + 1)):

                weights = tf.Variable(tf.truncated_normal([layers[idx], layers[idx+1]],
                                      stddev=1.0 / math.sqrt(float(layers[idx]))),
                                      name='weights')
                biases = tf.Variable(tf.zeros([layers[idx+1]]), name='biases')
                last = tf.nn.relu(tf.matmul(last, weights) + biases)

        # make output layer
        with tf.name_scope('out'):
            weights = tf.Variable(tf.truncated_normal([layers[idx], layers[idx+1]],
                                  stddev=1.0 / math.sqrt(float(layers[idx]))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([layers[idx+1]]), name='biases')
            out = tf.nn.relu(tf.matmul(last, weights) + biases)
    return out


def simple_lstm(input_tensor, layers, name):

    n_hidden_layers = len(layers) - 1
    last = input_tensor
    with tf.name_scope(name):
        # stack hidden layers
        for idx in range(n_hidden_layers):
            with tf.name_scope('layer' + str(idx + 1)):

                weights = tf.Variable(tf.truncated_normal([layers[idx], layers[idx+1]],
                                      stddev=1.0 / math.sqrt(float(layers[idx]))),
                                      name='weights')
                biases = tf.Variable(tf.zeros([layers[idx+1]]), name='biases')
                last = tf.nn.relu(tf.matmul(last, weights) + biases)

                tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)

    return last


def running_idx(start=0):
    a = start
    while True:
        yield a
        a += 1


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

