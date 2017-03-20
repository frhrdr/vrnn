from tensorflow.examples.tutorials.mnist import input_data
import os.path
import pickle
import time

import numpy as np
import tensorflow as tf

import vrnn_model as model
from utilities import NetGen


def get_sequential_mnist_batch_dict_generator(in_pl, hid_pl, eps_z, pd, data_dir='data/mnist/', stage='train'):
    if stage == 'train':
        data = input_data.read_data_sets(data_dir).train
    else:
        data = input_data.read_data_sets(data_dir).validation
    d = {}
    while True:
        x = np.reshape(data.next_batch(pd['batch_size'])[0], (pd['batch_size'], 28, 28))
        d[in_pl] = np.transpose(x, (1, 0, 2))
        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        yield d


def get_train_batch_dict_generator(data, in_pl, hid_pl, eps_z, pd):
    end_idx = 0
    d = {}
    while True:
        start_idx = end_idx
        end_idx += pd['batch_size']
        if end_idx < data.shape[1]:
            d[in_pl] = data[:, start_idx:end_idx, :]  # input
        else:
            d1 = data[:, start_idx:, :]
            d2 = data[:, :(end_idx % data.shape[1]), :]
            d[in_pl] = np.concatenate((d1, d2), axis=1)
            end_idx = end_idx % data.shape[1]

        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        yield d


def get_tracking_placeholders(pd):
    sub_losses = [tf.constant(0, dtype=tf.float32, name='kldiv_acc'),
                  tf.constant(0, dtype=tf.float32, name='log_p_acc'),
                  tf.constant(0, dtype=tf.float32, name='norm_acc'),
                  tf.constant(0, dtype=tf.float32, name='exp_acc'),
                  tf.constant(0, dtype=tf.float32, name='diff_acc')]
    if 'bin' in pd['model']:
        sub_losses.append(tf.constant(0, dtype=tf.float32, name='ce_loss_acc'))

    mean_0 = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='mean_prior_debug')
    cov_0 = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='cov_prior_debug')
    mean_z = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='mean_z_debug')
    cov_z = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['z_dim']], name='cov_z_debug')
    if 'gm' in pd['model']:
        mean_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['modes_out'], pd['x_dim']],
                             name='mean_x_debug')
        cov_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['modes_out'], pd['x_dim']],
                            name='cov_x_debug')
    else:
        mean_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['x_dim']], name='mean_x_debug')
        cov_x = tf.constant(0, dtype=tf.float32, shape=[pd['batch_size'], pd['x_dim']], name='cov_x_debug')
    dist_params = [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x]
    return [sub_losses, dist_params]


def run_training(pd):
    # make log directory and store param_dict
    if not os.path.exists(pd['log_path']):
        os.makedirs(pd['log_path'])
    pickle.dump(pd, open(pd['log_path'] + '/params.pkl', 'wb'))

    netgen = NetGen()
    nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
    for net in nets:
        netgen.add_net(pd[net])

    multi_input_nets = ['phi_enc', 'phi_dec']
    for net in multi_input_nets:
        netgen.weave_inputs(net)

    with tf.Graph().as_default():
        stop_fun = model.get_train_stop_fun(pd['seq_length'])
        loop_fun = model.get_train_loop_fun(pd, netgen.fd)

        in_pl = tf.placeholder(tf.float32, name='x_pl',
                               shape=(pd['seq_length'], pd['batch_size'], pd['in_dim']))
        eps_z = tf.placeholder(tf.float32, name='eps_z',
                               shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        err_acc = tf.constant(0, dtype=tf.float32, name='diff_acc')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        tracked_tensors = get_tracking_placeholders(pd)
        loop_vars = [in_pl, hid_pl, err_acc, count, f_state, eps_z, tracked_tensors]

        _ = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)

        bound_final = loop_res[2]
        sub_losses, dist_params = loop_res[-1]

        train_op = model.optimization(bound_final, pd['learning_rate'])

        if pd['train_data_path'] == 'load_mnist':
            train_dict = get_sequential_mnist_batch_dict_generator(in_pl, hid_pl, eps_z, pd, stage='train')
            valid_data = None
        else:
            train_data = np.load(pd['train_data_path'])
            valid_data = np.load(pd['valid_data_path'])
            train_dict = get_train_batch_dict_generator(train_data, in_pl, hid_pl, eps_z, pd)

        tf.summary.scalar('bound', bound_final)
        loss_names = ['kldiv', 'log_p', 'norm', 'exp', 'x_diff', 'bin_ce']
        for idx in range(len(sub_losses)):
            tf.summary.scalar(loss_names[idx], sub_losses[idx])

        dist_names = ['mean_0', 'cov_0', 'mean_z', 'cov_z', 'mean_x', 'cov_x']
        cuts = [40, 5, 40, 5, 10, 5]
        for t, name, cut in zip(dist_params, dist_names, cuts):
            tf.summary.histogram('debug/raw/' + name, t)
            t = tf.maximum(t, -cut)
            t = tf.minimum(t, cut)
            tf.summary.histogram('debug/cut/' + name, t)

        summary_op = tf.summary.merge_all()

        valid_bound = tf.summary.scalar('validation_bound', bound_final)

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(pd['log_path'] + '/summaries', sess.graph)
            start_time = time.time()

            if pd['load_path'] is None:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, pd['load_path'])

            saver = tf.train.Saver(max_to_keep=pd['num_ckpts'])

            for it in range(pd['max_iter']):
                feed = train_dict.next()

                _, err, summary_str = sess.run([train_op, bound_final, summary_op], feed_dict=feed)

                summary_writer.add_summary(summary_str, it)

                if (it + 1) % 100 == 0:
                    summary_writer.flush()

                if (pd['valid_freq'] > 0) and (it + 1) % pd['valid_freq'] == 0:
                    valid_dict = get_train_batch_dict_generator(valid_data, in_pl, hid_pl, eps_z, pd)
                    num_it = int(325 / pd['batch_size'])
                    err_acc = 0.0
                    for v_it in range(num_it):
                        feed = valid_dict.next()

                        _, err, summary_str = sess.run([train_op, bound_final, valid_bound], feed_dict=feed)
                        summary_writer.add_summary(summary_str, it)
                        err_acc += err
                    print('Iteration: ', it + 1, ' Validation Error: ', err_acc)

                if (pd['print_freq'] > 0) and (it + 1) % pd['print_freq'] == 0:

                    print('iteration ' + str(it + 1) +
                          ' error: ' + str(err) +
                          ' time: ' + str(time.time() - start_time))

                if (it + 1) % pd['log_freq'] == 0 or (it + 1) == pd['max_iter']:
                    checkpoint_file = os.path.join(pd['log_path'], 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(it + 1))


def get_gen_batch_dict_generator(hid_pl, eps_z, eps_x, pd):
    d = {}
    while True:
        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        d[eps_x] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['x_dim']))
        # d[eps_z] = np.zeros((pd['seq_length'], pd['batch_size'], pd['z_dim']))
        # d[eps_x] = np.zeros((pd['seq_length'], pd['batch_size'], pd['x_dim']))
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

        in_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['in_dim']], dtype=tf.float32)
        eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']),
                               name='eps_z')
        eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']),
                               name='eps_x')
        # if pd['model'] == 'gm_out':
        #     eps_pi = tf.placeholder(tf.int32, shape=(pd['seq_length'], pd['batch_size']))
        #     eps_out = [eps_x, eps_pi]
        # else:
        #     eps_out = eps_x
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)

        loop_vars = [in_pl, hid_pl, count, f_state, eps_z, eps_x]

        _ = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)
        x_final = loop_res[0]

        batch_dict = get_gen_batch_dict_generator(hid_pl, eps_z, eps_x, pd)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            feed = batch_dict.next()

            x_gen = sess.run(x_final, feed_dict=feed)

            return x_gen


def run_read_then_continue(params_file, read_seq, ckpt_file=None, batch_size=1):
    # build train model without actual train-op and run on inputs
    # retrieve last hidden state as well as lstm state
    # build gen model, init with saved states, run as often as desired
    # return generated sequences as array
    # only works with single lstm layer at this point

    pd = pickle.load(open(params_file, 'rb'))

    if ckpt_file is None:
        ckpt_file = pd['log_path'] + '/ckpt-' + str(pd['max_iter'])
    pd['batch_size'] = batch_size
    pd['seq_length'] = read_seq.shape[0]

    netgen = NetGen()
    nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
    for net in nets:
        netgen.add_net(pd[net])
    multi_input_nets = ['phi_enc', 'phi_dec']
    for net in multi_input_nets:
        netgen.weave_inputs(net)

    with tf.Graph().as_default():
        stop_fun = model.get_train_stop_fun(pd['seq_length'])
        loop_fun = model.get_train_loop_fun(pd, netgen.fd)

        in_pl = tf.placeholder(tf.float32, name='x_pl',
                               shape=(pd['seq_length'], pd['batch_size'], pd['in_dim']))
        eps_z = tf.placeholder(tf.float32, name='eps_z',
                               shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']))
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        err_acc = tf.constant(0, dtype=tf.float32, name='err_acc')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        tracked_tensors = get_tracking_placeholders(pd)
        loop_vars = [in_pl, hid_pl, err_acc, count, f_state, eps_z, tracked_tensors]

        loop_fun(*loop_vars)
        tf.get_variable_scope().reuse_variables()
        loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)

        # h_final = loop_res[1]
        f_final = loop_res[4]

        feed = {in_pl: read_seq,
                hid_pl: np.zeros((pd['batch_size'], pd['hid_state_size'])),
                eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim']))}

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)
            argin = list(f_final)
            res = sess.run(argin, feed_dict=feed)
            h = res[-1][1]

    # now that h and f are retrieved, build and run gen model
    with tf.Graph().as_default():
        stop_fun = model.get_gen_stop_fun(pd['seq_length'])
        loop_fun = model.get_gen_loop_fun(pd, netgen.fd)

        x_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['x_dim']], dtype=tf.float32)
        eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['z_dim']),
                               name='eps_z')
        eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['x_dim']),
                               name='eps_x')
        count = tf.constant(0, dtype=tf.float32, name='counter')
        h_state = tf.constant(h, name='ht_init')
        f_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.constant(k[0]), tf.constant(k[1])) for k in res])
        loop_vars = [x_pl, h_state, count, f_state, eps_z, eps_x]

        loop_fun(*loop_vars)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars)

        x_final = loop_res[0]

        feed = {eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['z_dim'])),
                eps_x: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['x_dim']))}

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            x_gen = sess.run(x_final, feed_dict=feed)

            return x_gen


# SUMMARIES
# tv = tf.trainable_variables()
# tv_summary = [tf.reduce_mean(k) for k in tv]
# tv_print = tf.Print(bound_final, tv_summary, message='tv ')

# for v in tv:
#     tf.summary.histogram('vars/' + v.name, v)
#
# grads = [g for g in tf.gradients(bound_final, tv) if g is not None]
# for g in grads:
#     name = g.name
#     tf.summary.histogram('grads/raw/' + name, g)
#     g = tf.maximum(g, -1000)
#     g = tf.minimum(g, 1000)
#     tf.summary.histogram('grads/cut1k/' + name, g)
#     g = tf.maximum(g, -10)
#     g = tf.minimum(g, 10)
#     tf.summary.histogram('grads/cut10/' + name, g)
