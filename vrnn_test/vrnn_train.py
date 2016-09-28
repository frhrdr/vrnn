import tensorflow as tf
from utilities import NetGen
import vrnn_model as model
# load param_dict for the overall model
from params import PARAM_DICT

# load the data


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
    # seems like x_list is better off as a +1d tensor
    # x_list = [tf.placeholder(tf.float32,
    #                          shape=(PARAM_DICT['batch_size'], PARAM_DICT['data_dim']),
    #                          name=('x_pl_t'+str(k)))
    #           for k in range(PARAM_DICT['seq_length'])]
    x_list = tf.placeholder(tf.float32, name='x_pl',
                            shape=(PARAM_DICT['seq_length'], PARAM_DICT['batch_size'], PARAM_DICT['data_dim']))
    hid_pl = tf.zeros([PARAM_DICT['batch_size'], PARAM_DICT['hid_state_size']], name='ht_init')
    err_acc = tf.zeros(1, name='err_acc')
    count = tf.to_int32(0, name='counter')
    loop_vars = [x_list, hid_pl, err_acc, count]
    # loop it
    loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                             parallel_iterations=1,
                             swap_memory=False,
                             name='while_loop')
    err_final = loop_res[3]
    # get the train_op
    train_op = model.train(err_final, PARAM_DICT['learning_rate'])
    # get a session
    with tf.Session() as sess:

        # run init variables op
        sess.run(tf.initialize_all_tables())

        for idx in range(PARAM_DICT['max_iter']):
            # fill feed_dict
            feed = {}

            # run train_op
            _, err = sess.run([err_acc, train_op], feed_dict=feed)
