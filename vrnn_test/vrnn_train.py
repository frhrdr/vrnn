import os.path
import pickle
import time

import numpy as np
import tensorflow as tf

import vrnn_model as model
from utilities import NetGen


# load param_dict for the overall model


def get_train_batch_dict_generator(data, x_pl, hid_pl, eps_z, pd):
    end_idx = 0
    d = {}
    while True:
        start_idx = end_idx
        end_idx += pd['batch_size']
        if end_idx < data.shape[1]:
            d[x_pl] = data[:, start_idx:end_idx, :]  # input
        else:
            d1 = data[:, start_idx:, :]
            d2 = data[:, :(end_idx % data.shape[1]), :]
            d[x_pl] = np.concatenate((d1, d2), axis=1)
            end_idx = end_idx % data.shape[1]

        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        # d[eps_z] = np.zeros((pd['seq_length'], pd['batch_size'], pd['z_dim']))  # for debugging
        yield d


def get_debug_pl(pd):
    mean_0 = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='mean_prior_debug')
    cov_0 = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='cov_prior_debug')
    mean_z = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='mean_z_debug')
    cov_z = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='cov_z_debug')
    if pd['model'] == 'gm_out':
        mean_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['modes_out'], pd['x_dim']], name='mean_x_debug')
        cov_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['modes_out'], pd['x_dim']], name='cov_x_debug')
    else:
        mean_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['x_dim']], name='mean_x_debug')
        cov_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['x_dim']], name='cov_x_debug')
    return [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x]


def run_training(pd):
    # make log directory and store param_dict
    if not os.path.exists(pd['log_path']):
        os.makedirs(pd['log_path'])
    pickle.dump(pd, open(pd['log_path'] + '/params.pkl', 'wb'))

    # load the data. expect numpy array of time_steps by samples by input dimension
    data = np.load(pd['data_path'])

    netgen = NetGen()
    nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
    for net in nets:
        netgen.add_net(pd[net])

    multi_input_nets = ['phi_enc', 'phi_dec']
    for net in multi_input_nets:
        netgen.weave_inputs(net)

    with tf.Graph().as_default() as graph:
        stop_fun = model.get_train_stop_fun(pd['seq_length'])
        loop_fun = model.get_train_loop_fun(pd, netgen.fd)

        x_pl = tf.placeholder(tf.float32, name='x_pl',
                              shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']))
        eps_z = tf.placeholder(tf.float32, name='eps_z',
                               shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        err_acc = [tf.constant(0, dtype=tf.float32, name='bound_acc'),
                   tf.constant(0, dtype=tf.float32, name='kldiv_acc'),
                   tf.constant(0, dtype=tf.float32, name='log_p_acc')]
        count = tf.constant(0, dtype=tf.float32, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        debug_tensors = get_debug_pl(pd)
        loop_vars = [x_pl, hid_pl, err_acc, count, f_state, eps_z, debug_tensors]

        loop_res = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)

        err_final = loop_res[2]
        bound_final = err_final[0]
        kldiv_final = err_final[1]
        log_p_final = err_final[2]

        # train_op, grad_print = model.train(bound_final, pd['learning_rate'])
        train_op = model.optimization(bound_final, pd['learning_rate'])

        batch_dict = get_train_batch_dict_generator(data, x_pl, hid_pl, eps_z, pd)

        # SUMMARIES
        tv = tf.trainable_variables()
        tv_summary = [tf.reduce_mean(k) for k in tv]
        # tv_print = tf.Print(bound_final, tv_summary, message='tv ')
        for v in tv:
            tf.summary.histogram('vars/' + v.name, v)

        grads = [g for g in tf.gradients(bound_final, tv) if g is not None]
        for g in grads:
            tf.summary.histogram('grads/' + g.name, g)
        tf.summary.scalar('bound', bound_final)
        tf.summary.scalar('kldiv', kldiv_final)
        tf.summary.scalar('log_p', log_p_final)

        debug = loop_res[-1]
        for t in debug:
            tf.summary.histogram('debug/' + t.name, t)

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(pd['log_path'] + '/summaries', sess.graph)
            start_time = time.time()
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()

            for it in range(pd['max_iter']):
                feed = batch_dict.next()

                _, err, summary_str = sess.run([train_op, bound_final, summary_op], feed_dict=feed)

                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()

                if (pd['print_freq'] > 0) and (it + 1) % pd['print_freq'] == 0:

                    print('iteration ' + str(it + 1) +
                          ' error: ' + str(err) +
                          ' time: ' + str(time.time() - start_time))

                    # DEBUG
                    # sess.run([grad_print, tv_print], feed_dict=feed)

                if (it + 1) % pd['log_freq'] == 0 or (it + 1) == pd['max_iter']:
                    checkpoint_file = os.path.join(pd['log_path'], 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(it + 1))


def get_gen_batch_dict_generator(hid_pl, eps_z, eps_out, pd):
    if pd['model'] == 'gauss_out':
        eps_x = eps_out
        eps_pi = None
    elif pd['model'] == 'gm_out':
        eps_x, eps_pi = eps_out
    else:
        raise NotImplementedError

    d = {}
    while True:
        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        d[eps_x] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['x_dim']))
        if eps_pi is not None:
            d[eps_pi] = np.random.randint(0, size=(pd['seq_length'], pd['batch_size']))
        yield d


def run_generation(params_file, ckpt_file=None, batch=None):

    pd = pickle.load(open(params_file, 'rb'))

    if ckpt_file is None:  # set default checkpoint file
        ckpt_file = pd['log_path'] + '/ckpt-' + str(pd['max_iter'])

    if batch is not None:  # needs testing
        pd['batch_size'] = batch

    netgen = NetGen()
    nets = ['phi_x', 'phi_prior', 'phi_z', 'phi_dec', 'f_theta']  # phi_enc is not used
    for net in nets:
        netgen.add_net(pd[net])

    netgen.weave_inputs('phi_dec')

    with tf.Graph().as_default():
        stop_fun = model.get_gen_stop_fun(pd['seq_length'])
        loop_fun = model.get_gen_loop_fun(pd, netgen.fd)

        x_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['x_dim']], dtype=tf.float32)
        eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']),
                               name='eps_z')
        eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']),
                               name='eps_x')
        if pd['model'] == 'gm_out':
            eps_z = tf.placeholder(tf.float32, shape=(pd['modes_out']))
            eps_out = [eps_x, eps_z]
        else:
            eps_out = eps_x
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        loop_vars = [x_pl, hid_pl, count, f_state, eps_z, eps_out]

        _ = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)
        x_final = loop_res[0]

        batch_dict = get_gen_batch_dict_generator(hid_pl, eps_z, eps_out, pd)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            feed = batch_dict.next()

            x_gen = sess.run(x_final, feed_dict=feed)

            return x_gen


# def run_read_then_continue(params_file, read_seq, ckpt_file=None, batch_size=1):  # seems nonsensical
#     # build train model without actual train-op and run on inputs
#     # retrieve last hidden state as well as lstm state
#     # build gen model, init with saved states, run as often as desired
#     # return generated sequences as array
#
#     pd = pickle.load(open(params_file, 'rb'))
#
#     if ckpt_file is None:
#         ckpt_file = pd['log_path'] + '/ckpt-' + str(pd['max_iter'])
#     pd['batch_size'] = batch_size
#     pd['seq_length'] = read_seq.shape[0]
#
#     netgen = NetGen()
#     nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
#     for net in nets:
#         netgen.add_net(pd[net])
#     multi_input_nets = ['phi_enc', 'phi_dec']
#     for net in multi_input_nets:
#         netgen.weave_inputs(net)
#
#     with tf.Graph().as_default():
#         stop_fun = model.get_train_stop_fun(pd['seq_length'])
#         loop_fun = model.get_train_loop_fun(pd, netgen.fd)
#
#         # x_pl = tf.placeholder(tf.float32, name='x_pl',
#         #                       shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']))
#         x_pl = tf.Variable(read_seq, name='x_pl')
#         eps_z = tf.placeholder(tf.float32, name='eps_z',
#                                shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
#         hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
#         err_acc = tf.Variable(0, dtype=tf.float32, trainable=False, name='err_acc')
#         count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')
#         f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
#         loop_vars = [x_pl, hid_pl, err_acc, count, f_state, eps_z]
#
#         loop_fun(*loop_vars)
#         tf.get_variable_scope().reuse_variables()
#         loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
#                                  parallel_iterations=1,
#                                  swap_memory=False)
#         h_final = loop_res[1]
#         f_final = loop_res[4]
#
#         feed = {x_pl: read_seq,
#                 hid_pl: np.zeros((pd['batch_size'], pd['hid_state_size'])),
#                 eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))}
#
#         with tf.Session() as sess:
#             saver = tf.train.Saver()
#             saver.restore(sess, ckpt_file)
#             res = sess.run([h_final] + f_final, feed_dict=feed)
#             h = res[0]
#             f = res[1:]
#
#     # now that h and f are retrieved, build and run gen model
#     with tf.Graph().as_default():
#         stop_fun = model.get_gen_stop_fun(pd['seq_length'])
#         loop_fun = model.get_gen_loop_fun(pd, netgen.fd)
#
#         x_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['x_dim']], dtype=tf.float32)
#         eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']),
#                                name='eps_z')
#         eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']),
#                                name='eps_x')
#         count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')  # tf.to_int32(0, name='counter')
#         hid_pl = tf.Variable(h, name='ht_init')
#         f_state = [tf.Variable(k) for k in f]
#         loop_vars = [x_pl, hid_pl, count, f_state, eps_z, eps_x]
#
#         loop_fun(*loop_vars)
#
#         with tf.variable_scope(tf.get_variable_scope(), reuse=True):
#             loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
#                                      parallel_iterations=1,
#                                      swap_memory=False)
#         x_final = loop_res[0]
#
#         feed = {eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim'])),
#                 eps_x: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['x_dim']))}
#
#         with tf.Session() as sess:
#             saver = tf.train.Saver()
#             saver.restore(sess, ckpt_file)
#
#             x_gen = sess.run(x_final, feed_dict=feed)
#
#             return x_gen
