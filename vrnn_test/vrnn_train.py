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

    # get the stop_fun
    stop_fun = model.get_stop_fun(PARAM_DICT['seq_length'])
    # get the loop_fun
    loop_fun = model.get_loop_fun(PARAM_DICT, netgen.fd)
    # define loop_vars: x_list, hid_pl, err_acc, count
    x_list = [tf.placeholder(tf.float32, shape=(PARAM_DICT['batch_size'], PARAM_DICT['data_dim']), name=('pl_t'+str(k)))
              for k in range(PARAM_DICT['seq_length'])]

    count = tf.Variable(0, trainable=False, dtype=tf.int32, name='counter')
    loop_vars = []
    # loop it
    loop_res = tf.while_loop(stop_fun, loop_fun, loop_vars,
                             parallel_iterations=1,
                             swap_memory=False,
                             name='while_loop')
    err_acc = loop_res[3]
    # get the train_op
    train_op = model.train(err_acc, PARAM_DICT['learning_rate'])
    # get a session
    with tf.Session() as sess:

        # run init variables op
        sess.run(tf.initialize_all_tables())

        for idx in range(PARAM_DICT['max_iter']):
            # fill feed_dict
            feed = {}

            # run train_op
            _, err = sess.run([err_acc, train_op], feed_dict=feed)
