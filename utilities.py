import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class NetGen:

    def __init__(self):
        self.fd = {}                    # function dict stores generator functions under name
        # self.vd = defaultdict(list)     # tensor dict stores list of associated variable tensors under function name
        # self.cells = {}                 # stores rnn cells by name (mostly for initialization)
        # self.states = {}                # stores rnn states by name

    def __str__(self):
        return str(self.fd.keys())

    # adds net-gen function of certain type to the internal dict

    def add_net(self, params):
        name = params['name']

        if params['nn_type'] == 'general_mlp':

            def f(in_pl):
                return general_mlp(in_pl, params)
            self.fd[name] = f

        if params['nn_type'] == 'simple_lstm':
            # given the odd way rnns are currently handled in tensorflow,
            # this function just creates an lstm cell which must then be called inside the loop
            # with the last state
            self.fd[name] = simple_lstm(params, name)

        if params['nn_type'] == 'general_lstm':
            self.fd[name] = general_lstm(params, name)

    # concatenates several tensors into one input to existing nn of given name
    def weave_inputs(self, name):
        f = self.fd[name]

        def g(*args):
            # maybe make dimension (1) flexible later
            in_pl = tf.concat(1, list(args), name=name + "_joint_inputs")
            return f(in_pl)

        self.fd[name] = g


def general_mlp(input_tensor, params):
    name = params['name']
    layers = params['layers']
    init_bias = 0
    if 'init_bias' in params:
        init_bias = params['init_bias']

    last = input_tensor
    with tf.name_scope(name):
        # stack layers
        for idx in range(1, len(layers)):

            weights = tf.get_variable(name=(name + '_w' + str(idx)),
                                      shape=[layers[idx-1], layers[idx]], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(name=(name + '_b' + str(idx)), dtype=tf.float32,
                                     initializer=(tf.zeros([layers[idx]]) + init_bias))

            if params['activation'] == 'relu':
                last = tf.nn.relu(tf.matmul(last, weights) + biases)
            else:  # default to tanh
                last = tf.nn.tanh(tf.matmul(last, weights) + biases)

    return last


def simple_lstm(params, name):
    n_units = params['layers']
    with tf.name_scope(name):
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_units)
    return cell


def general_lstm(params, name):
    layers = params['layers']
    cells = []
    with tf.name_scope(name):
        for layer in layers:
            cells.append(tf.nn.rnn_cell.LSTMCell(layer))
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    return multi_cell


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
