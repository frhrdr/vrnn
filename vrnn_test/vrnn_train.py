import tensorflow as tf
import numpy as np
from utilities import NetGen, get_batch_dict_gen
import vrnn_model as model
# load param_dict for the overall model
from params import PARAM_DICT


# load the data. expect numpy array of time_steps by samples by input dimension
data = np.load(PARAM_DICT['data_path'])


# make NetGen object
netgen = NetGen()
# use param_dict to add each required net_fun to the NetGen object
nets = ['phi_x', 'phi_prior', 'phi_enc', 'phi_z', 'phi_dec', 'f_theta']
for net in nets:
    netgen.add_net(PARAM_DICT[net])

# allow concatenation of multiple input tensors, where necessary
multi_input_nets = ['phi_enc', 'phi_dec', 'f_theta']
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
    loop_vars = [x_pl, hid_pl, err_acc, count]
    # loop it
    loop_dummy = loop_fun(*loop_vars)  # quick fix - need to init variables outside the loop
    loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                             parallel_iterations=1,
                             swap_memory=False,
                             name='while_loop')
    err_final = loop_res[2]
    # get the train_op
    train_op = model.train(err_final, PARAM_DICT['learning_rate'])

    # make a batch dict generator with the given placeholder
    batch_dict = get_batch_dict_gen(data, PARAM_DICT['num_batches'], x_pl, hid_pl,
                                    (PARAM_DICT['batch_size'], PARAM_DICT['hid_state_size']))

    # get a session
    with tf.Session() as sess:

        print(len(tf.all_variables()))
        # run init variables op
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        # sess.run(tf.initialize_all_tables())
        sess.run(init_op)
        for idx in range(PARAM_DICT['max_iter']):
            # fill feed_dict
            feed = batch_dict.next()

            # run train_op
            _, err = sess.run([err_acc, train_op], feed_dict=feed)
            print('iteration ' + str(idx) + ' done')
