import tensorflow as tf
import numpy as np
import time
import os.path
import pickle
from utilities import NetGen
import vrnn_model as model
# load param_dict for the overall model
from params import PARAM_DICT


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

        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))  # initial hidden state
        # 'fresh' noise for sampling
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['n_latent']))
        yield d


def run_training(param_dict):

    # for brevity
    pd = param_dict

    # make log directory and store param_dict
    if not os.path.exists(pd['log_path']):
        os.makedirs(pd['log_path'])
    pickle.dump(pd, open(pd['log_path'] + '/params.pkl', 'wb'))
    # set verbosity (doesn't seem to work)
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # load the data. expect numpy array of time_steps by samples by input dimension
    data = np.load(pd['data_path'])

    # make NetGen object
    netgen = NetGen()
    # use param_dict to add each required net_fun to the NetGen object
    nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
    for net in nets:
        netgen.add_net(pd[net])

    # allow concatenation of multiple input tensors, where necessary (f_theta is handled separately)
    multi_input_nets = ['phi_enc', 'phi_dec']
    for net in multi_input_nets:
        netgen.weave_inputs(net)

    # get a graph
    with tf.Graph().as_default():
        # get the stop condition and loop function
        stop_fun = model.get_train_stop_fun(pd['seq_length'])
        loop_fun = model.get_train_loop_fun(pd, netgen.fd, pd['watchlist'])

        # define loop_vars: x_list, hid_pl, err_acc, count
        x_pl = tf.placeholder(tf.float32, name='x_pl',
                              shape=(pd['seq_length'], pd['batch_size'], pd['data_dim']))
        eps_z = tf.placeholder(tf.float32, name='eps_z',
                               shape=(pd['seq_length'], pd['batch_size'], pd['n_latent']))
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        err_acc = tf.Variable(0, dtype=tf.float32, trainable=False, name='err_acc')
        count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')  # tf.to_int32(0, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        loop_vars = [x_pl, hid_pl, err_acc, count, f_state, eps_z]

        # loop it
        loop_res = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                                     parallel_iterations=1,  # can probably drop these params
                                     swap_memory=False)
        err_final = loop_res[2]

        # get the train_op
        train_op, grad_print = model.train(err_final, pd['learning_rate'])

        # make a batch dict generator with the given placeholder
        batch_dict = get_train_batch_dict_generator(data, x_pl, hid_pl, eps_z, pd)

        tv = tf.trainable_variables()
        tv_summary = [tf.reduce_mean(k) for k in tv]
        tv_print = tf.Print(err_acc, tv_summary, message='tv ')

        # get a session
        with tf.Session() as sess:

            # take start time
            start_time = time.time()

            # run init variables op
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()

            for it in range(pd['max_iter']):
                # fill feed_dict
                feed = batch_dict.next()

                # run train_op
                _, err = sess.run([train_op, err_final], feed_dict=feed)

                if (pd['print_freq'] > 0) and (it + 1) % pd['print_freq'] == 0:

                    print('iteration ' + str(it + 1) +
                          ' error: ' + str(err) +
                          ' time: ' + str(time.time() - start_time))

                    # DEBUG
                    sess.run([grad_print, tv_print], feed_dict=feed)

                # occasionally save weights and log
                if (it + 1) % pd['log_freq'] == 0 or (it + 1) == pd['max_iter']:
                    checkpoint_file = os.path.join(pd['log_path'], 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(it + 1))


def get_gen_batch_dict_generator(hid_pl, eps_z, eps_x, pd):
    d = {}
    while True:
        d[hid_pl] = np.zeros((pd['batch_size'], pd['hid_state_size']))  # initial hidden state
        # 'fresh' noise for sampling
        d[eps_z] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['n_latent']))
        d[eps_x] = np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['data_dim']))
        yield d


def run_generation(params_file, ckpt_file=None, batch=None):

    pd = pickle.load(open(params_file, 'rb'))

    # set default checkpoint file
    if ckpt_file is None:
        ckpt_file = pd['log_path'] + '/ckpt-' + str(pd['max_iter'])

    # set custom number of generated samples (not entirely sure this will work)
    if batch is not None:
        pd['batch_size'] = batch

    # make nets
    netgen = NetGen()
    nets = ['phi_x', 'phi_prior', 'phi_z', 'phi_dec', 'f_theta']  # phi_enc is not used
    for net in nets:
        netgen.add_net(pd[net])

    netgen.weave_inputs('phi_dec')

    with tf.Graph().as_default():
        # build gen model
        stop_fun = model.get_gen_stop_fun(pd['seq_length'])
        loop_fun = model.get_gen_loop_fun(pd, netgen.fd)

        x_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['data_dim']], dtype=tf.float32)
        eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['n_latent']),
                               name='eps_z')
        eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['data_dim']),
                               name='eps_x')
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')  # tf.to_int32(0, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        loop_vars = [x_pl, hid_pl, count, f_state, eps_z, eps_x]

        # loop it
        _ = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                                     parallel_iterations=1,
                                     swap_memory=False)
        x_final = loop_res[0]

        batch_dict = get_gen_batch_dict_generator(hid_pl, eps_z, eps_x, pd)
        print([(k.name, k.get_shape()) for k in tf.trainable_variables()])
        with tf.Session() as sess:
            # load weights
            saver = tf.train.Saver()
            saver.restore(sess, 'data/logs/handwriting/ckpt-500')
            # saver.restore(sess, ckpt_file)

            feed = batch_dict.next()
            # run generative model as desired
            x_gen = sess.run(x_final, feed_dict=feed)

            return x_gen


def run_read_then_continue(params_file, read_seq, ckpt_file=None, batch_size=1):
    # build train model without actual train-op and run on inputs
    # retrieve last hidden state as well as lstm state
    # build gen model, init with saved states, run as often as desired
    # return generated sequences as array

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
        loop_fun = model.get_train_loop_fun(pd, netgen.fd, pd['watchlist'])

        # x_pl = tf.placeholder(tf.float32, name='x_pl',
        #                       shape=(pd['seq_length'], pd['batch_size'], pd['data_dim']))
        x_pl = tf.Variable(read_seq, name='x_pl')
        eps_z = tf.placeholder(tf.float32, name='eps_z',
                               shape=(pd['seq_length'], pd['batch_size'], pd['n_latent']))
        hid_pl = tf.placeholder(tf.float32, shape=(pd['batch_size'], pd['hid_state_size']), name='ht_init')
        err_acc = tf.Variable(0, dtype=tf.float32, trainable=False, name='err_acc')
        count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')
        f_state = netgen.fd['f_theta'].zero_state(pd['batch_size'], tf.float32)
        loop_vars = [x_pl, hid_pl, err_acc, count, f_state, eps_z]

        loop_fun(*loop_vars)
        tf.get_variable_scope().reuse_variables()
        loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                                 parallel_iterations=1,
                                 swap_memory=False)
        h_final = loop_res[1]
        f_final = loop_res[4]

        feed = {x_pl: read_seq,
                hid_pl: np.zeros((pd['batch_size'], pd['hid_state_size'])),
                eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['n_latent']))}

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)
            res = sess.run([h_final] + f_final, feed_dict=feed)
            h = res[0]
            f = res[1:]

    # now that h and f are retrieved, build and run gen model
    with tf.Graph().as_default():
        stop_fun = model.get_gen_stop_fun(pd['seq_length'])
        loop_fun = model.get_gen_loop_fun(pd, netgen.fd)

        x_pl = tf.zeros([pd['seq_length'], pd['batch_size'], pd['data_dim']], dtype=tf.float32)
        eps_z = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['n_latent']),
                               name='eps_z')
        eps_x = tf.placeholder(tf.float32, shape=(pd['seq_length'], pd['batch_size'], pd['data_dim']),
                               name='eps_x')
        count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')  # tf.to_int32(0, name='counter')
        hid_pl = tf.Variable(h, name='ht_init')
        f_state = [tf.Variable(k) for k in f]
        loop_vars = [x_pl, hid_pl, count, f_state, eps_z, eps_x]

        loop_fun(*loop_vars)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                                     parallel_iterations=1,
                                     swap_memory=False)
        x_final = loop_res[0]

        feed = {eps_z: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['n_latent'])),
                eps_x: np.random.normal(size=(pd['seq_length'], pd['batch_size'], pd['data_dim']))}

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            x_gen = sess.run(x_final, feed_dict=feed)

            return x_gen
