import matplotlib
matplotlib.use('qt5agg', warn=False, force=True)
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

ACT_FUNCS = {'relu': tf.nn.relu,
             'tanh': tf.nn.tanh,
             'elu': tf.nn.elu}


class NetGen:
    """ Class for creating and storing network generator functions """
    def __init__(self):
        self.fd = {}                    # function dict stores generator functions under name

    def __str__(self):
        return str(self.fd.keys())

    def add_net(self, params):
        """ adds net-gen function of certain type to the internal dict """
        name = params['name']

        if params['nn_type'] == 'general_mlp':

            def f(in_pl):
                return general_mlp(in_pl, params)
            self.fd[name] = f

        if params['nn_type'] == 'simple_lstm':
            self.fd[name] = simple_lstm(params, name)

        if params['nn_type'] == 'general_lstm':
            self.fd[name] = general_lstm(params, name)

        if 'out2dist' in params:
            if params['out2dist'] == 'gauss':
                self.fd[name] = out_to_normal(self.fd[name], params)
            elif params['out2dist'] == 'gauss_plus_bin':
                self.fd[name] = out_to_normal_plus_binary(self.fd[name], params)
            elif params['out2dist'] == 'gm':
                self.fd[name] = out_to_gm(self.fd[name], params)
            elif params['out2dist'] == 'gm_plus_bin':
                self.fd[name] = out_to_gm_plus_binary(self.fd[name], params)

    def weave_inputs(self, name):
        """ concatenates several tensors into one input to existing nn of given name """
        f = self.fd[name]

        def g(*args):
            in_pl = tf.concat(list(args), axis=1, name=name + "_joint_inputs")
            return f(in_pl)

        self.fd[name] = g


def general_mlp(input_tensor, params):
    """ build a generic MLP as specified in params """
    name = params['name']
    layers = params['layers']
    init_bias = 0
    if 'init_bias' in params:
        init_bias = params['init_bias']

    last = input_tensor
    with tf.name_scope(name):
        for idx in range(1, len(layers)):

            weights = tf.get_variable(name=(name + '_w' + str(idx)),
                                      shape=[layers[idx-1], layers[idx]], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(name=(name + '_b' + str(idx)), dtype=tf.float32,
                                     initializer=(tf.random_normal([layers[idx]], mean=init_bias)))

            if 'activation' in params:
                act_key = params['activation']
            else:
                act_key = 'relu'  # default

            last = ACT_FUNCS[act_key](tf.matmul(last, weights) + biases)

    return last


def simple_lstm(params, name):
    """ build a one layer lstm"""
    n_units = params['layers']
    with tf.name_scope(name):
        cell = tf.contrib.rnn.BasicLSTMCell(n_units)
    return cell


def general_lstm(params, name):
    """ build an LSTM net as specified in params"""
    layers = params['layers']
    cells = []
    with tf.name_scope(name):
        for layer in layers:
            cells.append(tf.contrib.rnn.LSTMCell(layer))
    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
    return multi_cell


def out_to_normal(net_fun, params):
    """ adapt a net to output parameters of a decorrelated Gaussian distribution """
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)
            mean_weights = tf.get_variable(name + '_m_w', initializer=tf.random_normal([d_out, d_dist], mean=0))
            mean_biases = tf.get_variable(name + '_m_b', initializer=tf.random_normal([d_dist], mean=0))

            mean = tf.matmul(net_out, mean_weights) + mean_biases
            cov_weights = tf.get_variable(name + '_c_w', initializer=tf.random_normal([d_out, d_dist],
                                                                                      mean=0,
                                                                                      stddev=params['init_sig_var']))
            cov_biases = tf.get_variable(name + '_c_b', initializer=tf.random_normal([d_dist], mean=0))
            cov = tf.nn.softplus(tf.matmul(net_out, cov_weights) + cov_biases)
        return mean, cov
    return f


def out_to_normal_plus_binary(net_fun, params):
    """ adapt a net to output parameters of a decorrelated Gaussian distribution plus a single sigmoidal logit"""
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)
            mean_weights = tf.get_variable(name + '_m_w', initializer=tf.random_normal([d_out, d_dist], mean=0))
            mean_biases = tf.get_variable(name + '_m_b', initializer=tf.random_normal([d_dist], mean=0))

            mean = tf.matmul(net_out, mean_weights) + mean_biases
            cov_weights = tf.get_variable(name + '_c_w', initializer=tf.random_normal([d_out, d_dist],
                                                                                      mean=0,
                                                                                      stddev=params['init_sig_var']))
            cov_biases = tf.get_variable(name + '_c_b', initializer=tf.random_normal([d_dist], mean=0))
            cov = tf.nn.softplus(tf.matmul(net_out, cov_weights) + cov_biases)

            bin_weights = tf.get_variable(name + '_bin_w', initializer=tf.random_normal([d_out, 1], mean=0, stddev=0.01))
            bin_biases = tf.get_variable(name + '_bin_b', initializer=tf.random_normal([1], mean=0))
            logits = tf.matmul(net_out, bin_weights) + bin_biases
        return mean, cov, logits
    return f


def out_to_gm(net_fun, params):
    """ adapt a net to output parameters of a decorrelated Gaussian mixture distribution """
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    num_modes = params['modes']
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)

            pi_weights = tf.get_variable(name + '_pi', initializer=tf.random_normal([d_out, num_modes], mean=0))
            pi_logit = tf.matmul(net_out, pi_weights)

            mean_weights = tf.get_variable(name + '_m',
                                           initializer=tf.random_normal([d_out, num_modes * d_dist], mean=0))
            mean = tf.reshape(tf.matmul(net_out, mean_weights), [-1, num_modes, d_dist])

            cov_weights = tf.get_variable(name + '_c',
                                          initializer=tf.random_normal([d_out, num_modes * d_dist],
                                                                       mean=params['init_sig_bias'],
                                                                       stddev=params['init_sig_var']))
            cov = tf.reshape(tf.nn.softplus(tf.matmul(net_out, cov_weights)), [-1, num_modes, d_dist])

        return mean, cov, pi_logit
    return f


def out_to_gm_plus_binary(net_fun, params):
    """ adapt a net to output parameters of a decorrelated Gaussian mixture plus a single sigmoidal logit """
    d_dist = params['dist_dim']
    d_out = params['layers'][-1]
    num_modes = params['modes']
    name = params['name']

    def f(in_pl):
        with tf.name_scope(name):
            net_out = net_fun(in_pl)

            pi_weights = tf.get_variable(name + '_pi_w', initializer=tf.random_normal([d_out, num_modes], mean=0))
            pi_biases = tf.get_variable(name + '_pi_b', initializer=tf.random_normal([num_modes], mean=0))
            pi_logit = tf.matmul(net_out, pi_weights) + pi_biases

            mean_weights = tf.get_variable(name + '_m_w',
                                           initializer=tf.random_normal([d_out, num_modes * d_dist], mean=0))
            mean_biases = tf.get_variable(name + '_m_b', initializer=tf.random_normal([num_modes * d_dist], mean=0))
            mean = tf.reshape(tf.matmul(net_out, mean_weights) + mean_biases, [-1, num_modes, d_dist])

            cov_weights = tf.get_variable(name + '_c_w',
                                          initializer=tf.random_normal([d_out, num_modes * d_dist],
                                                                       mean=params['init_sig_bias'],
                                                                       stddev=params['init_sig_var']))
            cov_biases = tf.get_variable(name + '_c_b', initializer=tf.random_normal([num_modes * d_dist], mean=0))
            cov = tf.reshape(tf.nn.softplus(tf.matmul(net_out, cov_weights) + cov_biases), [-1, num_modes, d_dist])

            bin_weights = tf.get_variable(name + '_bin_w', initializer=tf.random_normal([d_out, 1], mean=0, stddev=0.01))
            bin_biases = tf.get_variable(name + '_bin_b', initializer=tf.random_normal([1], mean=0, stddev=0.01))
            bin_logit = tf.matmul(net_out, bin_weights) + bin_biases
        return mean, cov, pi_logit, bin_logit
    return f


# SOME BASIC UTILITY FUNCTIONS BELOW

def running_idx(start=0):
    a = start
    while True:
        yield a
        a += 1


def plot_img_mats(mat):
    """ plot l*m*n mats as l m by n gray-scale images """
    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n // cols))
    mat = np.maximum(mat, 0.0)
    mat = np.minimum(mat, 1.0)
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


def count_params(param_dict, net_names):
    """ compute rough estimate of number of params in a model based on its networks """
    num_params = 0
    nets = [param_dict[k] for k in net_names]
    for net in nets:
        l = net['layers']
        for idx in range(1, len(l)):
            num_params += (l[idx-1] + 1) * l[idx]
        if 'out2dist' in net.keys():
            # assume all gauss
            d = net['dist_dim']
            num_params += 2 * (l[-1] + 1) * d

    print('Architecure with about ' + str(num_params) + ' parameters')
