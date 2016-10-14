import tensorflow as tf
import numpy as np
import time
from utilities import NetGen, get_batch_dict_gen
import vrnn_model as model
# load param_dict for the overall model
from params import PARAM_DICT

# set verbosity (doesn't seem to work)
# tf.logging.set_verbosity(tf.logging.ERROR)

# load the data. expect numpy array of time_steps by samples by input dimension
data = np.load(PARAM_DICT['data_path'])


# make NetGen object
netgen = NetGen()
# use param_dict to add each required net_fun to the NetGen object
nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
for net in nets:
    netgen.add_net(PARAM_DICT[net])

# allow concatenation of multiple input tensors, where necessary (f_theta is handled separately)
multi_input_nets = ['phi_enc', 'phi_dec']
for net in multi_input_nets:
    netgen.weave_inputs(net)


# get a graph
with tf.Graph().as_default():

    # get the stop condition and loop function
    stop_fun = model.get_stop_fun(PARAM_DICT['seq_length'])
    loop_fun = model.get_loop_fun(PARAM_DICT, netgen.fd)

    # define loop_vars: x_list, hid_pl, err_acc, count
    x_pl = tf.placeholder(tf.float32, name='x_pl',
                          shape=(PARAM_DICT['seq_length'], PARAM_DICT['batch_size'], PARAM_DICT['data_dim']))
    hid_pl = tf.placeholder(tf.float32, shape=(PARAM_DICT['batch_size'], PARAM_DICT['hid_state_size']), name='ht_init')
    err_acc = tf.Variable(0, dtype=tf.float32, trainable=False, name='err_acc')
    count = tf.Variable(0, dtype=tf.float32, trainable=False, name='counter')  # tf.to_int32(0, name='counter')
    f_state = netgen.fd['f_theta'].zero_state(PARAM_DICT['batch_size'], tf.float32)
    # f_state = tf.Variable(0, dtype=tf.float32, trainable=False, name='debug') # placeholder for case without lstm
    loop_vars = [x_pl, hid_pl, err_acc, count, f_state]
    # loop it
    loop_dummy = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop
    tf.get_variable_scope().reuse_variables()  # quick fix - only needed for rnn. no idea why
    loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                             parallel_iterations=1,
                             swap_memory=False,
                             name='while_loop')
    err_final = loop_res[2]
    count_final = loop_res[3]
    # get the train_op
    train_op = model.train(err_final, PARAM_DICT['learning_rate'])

    # make a batch dict generator with the given placeholder
    batch_dict = get_batch_dict_gen(data, PARAM_DICT['num_batches'], x_pl, hid_pl,
                                    (PARAM_DICT['batch_size'], PARAM_DICT['hid_state_size']))

    # get a session
    with tf.Session() as sess:

        # take start time
        start_time = time.time()

        print("number of variables: " + str(len(tf.all_variables())))
        # run init variables op
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)

        # print any other tracked variables in the loop
        netweights = [netgen.vd['phi_z'][0], netgen.vd['phi_x'][0], netgen.vd['phi_enc'][0],
                      netgen.vd['phi_dec'][0], netgen.vd['phi_prior'][0]]
        # netgen.vd['f_theta'][0]
        err_final = tf.Print(err_final, netweights, message='netweights ', summarize=1)
        # err_final = tf.Print(err_final, err_final, message='ok_list')
        # common feature: all broken nets backprop through h
        # without loop, only phi_x and phi_enc break, suggesting the source is at enc, or rather KLdiv / z

        for idx in range(PARAM_DICT['max_iter']):
            # fill feed_dict
            feed = batch_dict.next()

            # make SummaryWriter


            # run train_op
            _, err = sess.run([train_op, err_final], feed_dict=feed)

            print('iteration ' + str(idx) + ' error: ' + str(err) + ' time: ' + str(time.time() - start_time))

            # occasionally save weights and log

            # wait a sec for debug messages
            time.sleep(1)
