import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from vrnn_model import gaussian_log_p, get_train_stop_fun, optimization
from utilities import NetGen
import os.path
import time
import pickle

PARAMS = dict()

PARAMS['max_iter'] = 20000
PARAMS['log_path'] = 'data/logs/ref_lstm_01/'
PARAMS['log_freq'] = 500
PARAMS['print_freq'] = 500
PARAMS['valid_freq'] = 500
PARAMS['load_path'] = None  # 'data/logs/mnist_16/ckpt-20000'
PARAMS['validation_set_size'] = 10000

PARAMS['model'] = 'gauss_out'  # options: gauss_out, gm_out, gauss_out_bin, gm_out_bin
PARAMS['modes_out'] = 1
PARAMS['x_dim'] = 28
PARAMS['lstm_dim'] = 2000
PARAMS['hid_mlp_dim'] = 500
PARAMS['batch_size'] = 100
PARAMS['x_dim'] = 28
PARAMS['seq_length'] = 28
PARAMS['learning_rate'] = 0.0001
PARAMS['validation_set_size'] = 1000

PARAMS['in_mlp'] = {'name': 'in_mlp',
                    'nn_type': 'general_mlp',
                    'activation': 'relu',
                    'layers': [PARAMS['x_dim'], PARAMS['hid_mlp_dim'], PARAMS['lstm_dim']],
                    'init_sig_var': 0.01,
                    'init_sig_bias': 0.0}
PARAMS['lstm'] = {'name': 'lstm',
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
    err_t = tf.reduce_mean(gaussian_log_p(dist, x_next, params['x_dim'])[0])
    err_acc -= err_t
    count += 1
    return x_pl, state, err_acc, count


def get_lstm_loop_fun(params, fd):
    def lstm_loop_fun(in_pl, state, err_acc, count):
        return lstm_loop(in_pl, state, err_acc, count, params, fd)
    return lstm_loop_fun


def get_sequential_mnist_batch_dict_generator(in_pl, params, data_dir='data/mnist/', stage='train'):
    if stage == 'train':
        data = input_data.read_data_sets(data_dir).train
    else:
        data = input_data.read_data_sets(data_dir).validation
    d = {}
    while True:
        x = np.reshape(data.next_batch(params['batch_size'])[0], (params['batch_size'], 28, 28))
        d[in_pl] = np.transpose(x, (1, 0, 2))
        yield d


def lstm_train(params):
    if not os.path.exists(params['log_path']):
        os.makedirs(params['log_path'])
    pickle.dump(params, open(params['log_path'] + '/params.pkl', 'wb'))

    netgen = NetGen()
    nets = ['in_mlp', 'lstm', 'out_mlp']
    for net in nets:
        netgen.add_net(params[net])

    with tf.Graph().as_default():
        stop_fun = get_train_stop_fun(params['seq_length'] - 1)
        loop_fun = get_lstm_loop_fun(params, netgen.fd)

        in_pl = tf.placeholder(tf.float32, name='x_pl',
                               shape=(params['seq_length'], params['batch_size'], params['x_dim']))
        err_acc = tf.constant(0, dtype=tf.float32, name='diff_acc')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        state = netgen.fd['lstm'].zero_state(params['batch_size'], tf.float32)
        loop_vars = [in_pl, state, err_acc, count]

        _ = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)

        bound_final = loop_res[2]

        train_op = optimization(bound_final, params['learning_rate'])

        train_dict = get_sequential_mnist_batch_dict_generator(in_pl, params, stage='train')
        valid_dict = get_sequential_mnist_batch_dict_generator(in_pl, params, stage='validation')

        tf.summary.scalar('bound', bound_final)
        summary_op = tf.summary.merge_all()

        valid_bound = tf.summary.scalar('validation_bound', bound_final)

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(params['log_path'] + '/summaries', sess.graph)
            start_time = time.time()

            if params['load_path'] is None:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, params['load_path'])

            saver = tf.train.Saver()

            for it in range(params['max_iter']):
                feed = next(train_dict)

                _, err, summary_str = sess.run([train_op, bound_final, summary_op], feed_dict=feed)

                summary_writer.add_summary(summary_str, it)

                if (it + 1) % 100 == 0:
                    summary_writer.flush()

                if (params['valid_freq'] > 0) and (it + 1) % params['valid_freq'] == 0:
                    num_it = int(params['validation_set_size'] / params['batch_size'])
                    err_acc = 0.0
                    for v_it in range(num_it):
                        feed = next(valid_dict)

                        _, err, summary_str = sess.run([train_op, bound_final, valid_bound], feed_dict=feed)
                        summary_writer.add_summary(summary_str, it)
                        err_acc += err
                    print('Iteration: ', it + 1, ' Validation Error: ', err_acc / params['validation_set_size'])

                if (params['print_freq'] > 0) and (it + 1) % params['print_freq'] == 0:

                    print('iteration ' + str(it + 1) +
                          ' error: ' + str(err) +
                          ' time: ' + str(time.time() - start_time))

                if (it + 1) % params['log_freq'] == 0 or (it + 1) == params['max_iter']:
                    checkpoint_file = os.path.join(params['log_path'], 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(it + 1))


lstm_train(PARAMS)
