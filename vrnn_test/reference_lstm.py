import tensorflow as tf
from vrnn_model import gaussian_log_p, get_train_stop_fun
from utilities import NetGen
import os
import pickle

PARAMS = {}

PARAMS['x_dim'] = 28
PARAMS['lstm_dim'] = 2000
PARAMS['hid_mlp_dim'] = 500
PARAMS['batch_size'] = 100
PARAMS['x_dim'] = 28
PARAMS['seq_length'] = 28
PARAMS['learning_rate'] = 0.0001
PARAMS['max_iter'] = 20000

PARAMS['in_mlp'] = {'name': 'in_mlp',
                     'nn_type': 'general_mlp',
                     'activation': 'relu',
                     'layers': [PARAMS['x_dim'], PARAMS['hid_mlp_dim'], PARAMS['lstm_dim']],
                     'init_sig_var': 0.01,
                     'init_sig_bias': 0.0}
PARAMS['lstm'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [PARAMS['lstm_dim'], PARAMS['lstm_dim']]}
PARAMS['out_mlp'] = {'name': 'out_mlp',
                     'nn_type': 'general_mlp',
                     'activation': 'relu',
                     'layers': [PARAMS['lstm_dim'], PARAMS['hid_mlp_dim']],
                     'out2dist': 'gauss',
                     'init_sig_var': 0.01,
                     'init_sig_bias': 0.0,
                     'dist_dim': PARAMS['x_dim']}


def lstm_inference(x_pl, state, fd):
    a = fd['in_mlp'](x_pl)
    b, state = fd['lstm'](a, state)
    dist = fd['out_mlp'](b)
    return dist, state


def lstm_loop(x_pl, state, err_acc, count, params, fd):
    x_now = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1]), axis=[0])
    x_next = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count + 1), 0, 0], [1, -1, -1]), axis=[0])
    dist, state = lstm_inference(x_now, state, fd)
    err_t = gaussian_log_p(dist, x_next, params['x_dim'])
    err_acc -= err_t
    count += 1
    return x_pl, state, err_acc, count


def get_lstm_loop_fun(params, fd):
    def lstm_loop_fun(in_pl, state, err_acc, count):
        return lstm_loop(in_pl, state, err_acc, count, params, fd)
    return lstm_loop_fun


def lstm_train(params):
    if not os.path.exists(params['log_path']):
        os.makedirs(params['log_path'])
    pickle.dump(params, open(params['log_path'] + '/params.pkl', 'wb'))

    netgen = NetGen()
    nets = ['in_mlp', 'lstm', 'out_mlp']
    for net in nets:
        netgen.add_net(params[net])


