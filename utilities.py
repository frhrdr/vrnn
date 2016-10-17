import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class NetGen:

    def __init__(self):
        self.fd = {}                    # function dict stores generator functions under name
        self.vd = defaultdict(list)     # tensor dict stores list of associated variable tensors under function name
        # self.cells = {}                 # stores rnn cells by name (mostly for initialization)
        # self.states = {}                # stores rnn states by name

    def __str__(self):
        return str(self.fd.keys())

    # adds net-gen function of certain type to the internal dict

    def add_net(self, params):
        name = params['name']
        var_list = self.vd[name]

        if params['nn_type'] == 'simple_mlp':
            n_in, n_hid, n_out = params['layers']

            def f(in_pl):
                return simple_mlp(in_pl, n_in, n_hid, n_out, name, var_list)
            self.fd[name] = f

        if params['nn_type'] == 'general_mlp':
            layers = params['layers']

            def f(in_pl):
                return general_mlp(in_pl, layers, name, var_list)
            self.fd[name] = f

        if params['nn_type'] == 'simple_lstm':
            # given the odd way rnns are currently handled in tensorflow,
            # this function just creates an lstm cell which must then be called inside the loop
            # with the last state
            layers = params['layers']
            self.fd[name] = simple_lstm(layers, name)

        if params['nn_type'] == 'general_lstm':
            layers = params['layers']
            self.fd[name] = general_lstm(layers, name)

    # concatenates several tensors into one input to existing nn of given name
    def weave_inputs(self, name):
        f = self.fd[name]

        def g(*args):
            # maybe make dimension (1) flexible later
            in_pl = tf.concat(1, list(args), name=name + "_joint_inputs")
            return f(in_pl)

        self.fd[name] = g

    def clear_var_dict(self):
        for val in self.vd.values():
            val[:] = []

    def init_rnn_states(self):
        pass


def simple_mlp(input_tensor, n_in, n_hid, n_out, scope, var_list):

    ridx = running_idx()
    make_vars = var_list == []

    # hidden layer
    with tf.name_scope(scope):
        with tf.name_scope('hidden'):
            if make_vars:
                var_list.append(tf.Variable(tf.truncated_normal([n_in, n_hid],
                                            stddev=1.0 / math.sqrt(float(n_in))),
                                            name='weights'))
            # print(make_vars)
            # print(var_list)
            weights = var_list[ridx.next()]

            if make_vars:
                var_list.append(tf.Variable(tf.zeros([n_hid]), name='biases'))
            biases = var_list[ridx.next()]

            hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        # output layer
        with tf.name_scope('output'):
            if make_vars:
                var_list.append(tf.Variable(tf.truncated_normal([n_hid, n_out],
                                            stddev=1.0 / math.sqrt(float(n_in))),
                                            name='weights'))
            weights = var_list[ridx.next()]

            if make_vars:
                var_list.append(tf.Variable(tf.zeros([n_out]), name='biases'))
            biases = var_list[ridx.next()]

            out_tensor = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    return out_tensor


def general_mlp(input_tensor, layers, name, var_list):

    ridx = running_idx()
    make_vars = var_list == []

    n_hidden_layers = len(layers) - 2
    last = input_tensor
    with tf.name_scope(name):
        # stack hidden layers
        for idx in range(n_hidden_layers):
            with tf.name_scope('hidden' + str(idx + 1)):
                if make_vars:
                    var_list.append(tf.Variable(tf.truncated_normal([layers[idx], layers[idx+1]],
                                                stddev=1.0 / math.sqrt(float(layers[idx]))),
                                                name='weights'))
                weights = var_list[ridx.next()]

                if make_vars:
                    var_list.append(tf.Variable(tf.zeros([layers[idx+1]]), name='biases'))
                biases = var_list[ridx.next()]

                last = tf.nn.relu(tf.matmul(last, weights) + biases)

        # make output layer
        with tf.name_scope('out'):
            if make_vars:
                var_list.append(tf.Variable(tf.truncated_normal([layers[-2], layers[-1]],
                                            stddev=1.0 / math.sqrt(float(layers[-2]))),
                                            name='weights'))
            weights = var_list[ridx.next()]

            if make_vars:
                var_list.append(tf.Variable(tf.zeros([layers[-1]]), name='biases'))
            biases = var_list[ridx.next()]

            out = tf.nn.relu(tf.matmul(last, weights) + biases)
    return out


def simple_lstm(n_units, name):
    with tf.name_scope(name):
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_units)
    return cell


def general_lstm(layers, name):
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
