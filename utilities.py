import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

ACT_FUNCS = {'relu': tf.nn.relu,
             'tanh': tf.nn.tanh,
             'elu': tf.nn.elu
            }

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

        if 'out2normal' in params and params['out2normal'] == True:
            self.fd[name] = out_to_normal(self.fd[name], params)

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
                                     initializer=(tf.random_normal([layers[idx]], mean=init_bias)))

            act_key = 'relu'  # default
            if 'activation' in params:
                act_key = params['activation']

            last = ACT_FUNCS[act_key](tf.matmul(last, weights) + biases)

            if 'use_batch_norm' in params and params['use_batch_norm'] == True:
                # bn_mean = tf.get_variable(name=(name + '_bn_mean' + str(idx)), dtype=tf.float32,
                #                           initializer=(tf.random_normal([layers[idx]], mean=init_bias)))
                # bn_var = tf.get_variable(name=(name + '_bn_var' + str(idx)), dtype=tf.float32,
                #                          initializer=(tf.random_normal([layers[idx]], mean=init_bias)))
                # offset = None
                # scale = None
                # eps = 0.00001
                # last = tf.nn.batch_normalization(last, bn_mean, bn_var, offset, scale, eps)
                last = tf.contrib.layers.batch_norm(last)
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


def out_to_normal(net_fun, params):
    dims = params['layers'][-1] / 2
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)
            out_m = tf.slice(net_out, [0, 0], [-1, dims])
            out_c = tf.slice(net_out, [0, dims], [-1, dims])
            mean_weights = tf.get_variable(name + '_m', initializer=tf.random_normal([dims, dims], mean=0))
            mean = tf.matmul(out_m, mean_weights)
            # mean = tf.Print(mean, [mean, out_m, mean_weights, out_c], message=name + ' m ')
            # mean = out_m
            cov_weights = tf.get_variable(name + '_c', initializer=tf.random_normal([dims, dims], mean=0, stddev=0.01))
            cov = tf.nn.softplus(tf.matmul(out_c, cov_weights))
            # cov = tf.exp(tf.matmul(out_c, cov_weights))
            # cov = tf.nn.softplus(out_c)
            # cov = tf.exp(out_c)
            # cov = tf.Print(cov, [cov], message=name + '_c ')
            return mean, cov
    return f


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
