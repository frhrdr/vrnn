import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tensorflow.contrib.slim import batch_norm

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

        if 'out2dist' in params:
            if params['out2dist'] == 'normal':
                self.fd[name] = out_to_normal(self.fd[name], params)
            if params['out2dist'] == 'normal_split':
                self.fd[name] = out_to_normal_split(self.fd[name], params)
            elif params['out2dist'] == 'normal_plus_binary':
                self.fd[name] = out_to_normal_plus_binary(self.fd[name], params)
            elif params['out2dist'] == 'gmm':
                self.fd[name] = out_to_gm(self.fd[name], params)

    # concatenates several tensors into one input to existing nn of given name
    def weave_inputs(self, name):
        f = self.fd[name]

        def g(*args):
            # maybe make dimension (1) flexible later
            in_pl = tf.concat(list(args), axis=1, name=name + "_joint_inputs")
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

            # tf.summary.histogram(name + '_w', weights)  # decided to do this centrally for all variables
            # tf.summary.histogram(name + '_b', biases)

            act_key = 'relu'  # default
            if 'activation' in params:
                act_key = params['activation']

            last = ACT_FUNCS[act_key](tf.matmul(last, weights) + biases)

            if 'use_batch_norm' in params and params['use_batch_norm'] == True:
                last = batch_norm(last,
                                  decay=0.999,
                                  center=True,
                                  scale=False,
                                  epsilon=0.001,
                                  moving_vars='moving_vars',
                                  activation=None,
                                  is_training=True,  # this actually necessitates a larger reworking.
                                  trainable=True,
                                  restore=True,
                                  scope=None,
                                  reuse=None)
    return last


def simple_lstm(params, name):
    n_units = params['layers']
    with tf.name_scope(name):
        cell = tf.contrib.rnn.BasicLSTMCell(n_units)
    return cell


def general_lstm(params, name):
    layers = params['layers']
    cells = []
    with tf.name_scope(name):
        for layer in layers:
            cells.append(tf.contrib.rnn.LSTMCell(layer))
    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
    return multi_cell


def out_to_normal(net_fun, params):
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)
            mean_weights = tf.get_variable(name + '_m', initializer=tf.random_normal([d_out, d_dist], mean=0))
            mean = tf.matmul(net_out, mean_weights)
            cov_weights = tf.get_variable(name + '_c', initializer=tf.random_normal([d_out, d_dist], mean=0, stddev=0.01))
            cov = tf.nn.softplus(tf.matmul(net_out, cov_weights), name=name + '_softplus')
            cov = cov + tf.constant(0.0001, dtype=tf.float32, name='min_variance')
        return mean, cov
    return f


def out_to_normal_split(net_fun, params):  # now old
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


def out_to_normal_plus_binary(net_fun, params):
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)
            mean_weights = tf.get_variable(name + '_m', initializer=tf.random_normal([d_out, d_dist], mean=0))
            mean = tf.matmul(net_out, mean_weights)
            cov_weights = tf.get_variable(name + '_c', initializer=tf.random_normal([d_out, d_dist], mean=0, stddev=0.01))
            cov = tf.nn.softplus(tf.matmul(net_out, cov_weights))
            bin_weights = tf.get_variable(name + '_bin', initializer=tf.random_normal([d_out, 1], mean=0, stddev=0.01))
            binary = tf.nn.tanh(tf.matmul(net_out, bin_weights))
        return mean, cov, binary
    return f


def out_to_gm(net_fun, params):
    out_size = params['layers'][-1]
    name = params['name']
    num_splits = params['splits']
    dims = int(out_size / (2 * num_splits) - 0.5)

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)

            # slice output into [mean, cov, pi] chunks
            means = []
            covs = []
            pis = []
            for split in range(num_splits):
                # each iteration uses 2*dims+1 outputs
                offset = (2*dims+1) * split
                out_m = tf.slice(net_out, [0, offset], [-1, dims])
                out_c = tf.slice(net_out, [0, offset + dims], [-1, dims])
                out_p = tf.slice(net_out, [0, offset + dims + 1], [-1, 1])
                mean_weights = tf.get_variable(name + '_m', initializer=tf.random_normal([dims, dims], mean=0))
                mean = tf.matmul(out_m, mean_weights)
                means.append(mean)

                cov_weights = tf.get_variable(name + '_c', initializer=tf.random_normal([dims, dims], mean=0, stddev=0.01))
                cov = tf.nn.softplus(tf.matmul(out_c, cov_weights))
                covs.append(cov)

                pi = tf.nn.softplus(out_p)
                pis.append(pi)
        return means, covs, pis
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