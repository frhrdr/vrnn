import tensorflow as tf
import math


def simple_mlp(input_tensor, n_in, n_hid, n_out):
    # hidden layer
    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([n_in, n_hid],
                                stddev=1.0 / math.sqrt(float(n_in))),
            name='weights')
        biases = tf.Variable(tf.zeros([n_hid]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # output layer
    with tf.name_scope('output'):
        weights = tf.Variable(
            tf.truncated_normal([n_hid, n_out],
                                stddev=1.0 / math.sqrt(float(n_hid))),
            name='weights')
        biases = tf.Variable(tf.zeros([n_out]),
                             name='biases')
        out_tensor = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    return out_tensor


def inference(images, batch_size, encoder_nn=None, decoder_nn=None, n_latentvars=10):

    # make encoder if not provided
    if encoder_nn is None:
        with tf.name_scope('encoder'):
            encoder_nn = simple_mlp(images, 28**2, 128, 2*n_latentvars)

    # create mean and covariance

    # batch_size = encoder_nn.get_shape() (How to do this properly??)
    mean_vec = tf.slice(encoder_nn, [0, 0], [batch_size, n_latentvars], name='mu')
    cov_vec = tf.slice(encoder_nn, [0, n_latentvars], [batch_size, n_latentvars], name='Sigma')
    # cov_mat = tf.batch_matrix_diag(cov_vec)

    # draw samples
    noise_vec = tf.random_normal([n_latentvars], mean=0.0, stddev=1.0, dtype=tf.float32, seed=123, name='z_noise')
    z_vec = mean_vec + tf.mul(cov_vec, noise_vec)

    # make decoder if not provided
    if decoder_nn is None:
        with tf.name_scope('decoder'):
            decoder_nn = simple_mlp(z_vec, n_latentvars, 128, 28**2)

    return decoder_nn, mean_vec, cov_vec


def loss(vae, mean_vec, cov_vec, target, n_latentvars):

    # KL divergence (see VAE Tut formula 7)
    # trace and determinant could be optimized.
    # k = tf.to_double(mean_vec.get_shape[0]) (How??)
    k = tf.to_float(n_latentvars)
    cov_trace = tf.reduce_sum(cov_vec, reduction_indices=[1], name='l1')
    cov_determinant = tf.reduce_prod(cov_vec, reduction_indices=[1], name='l2')
    mean_squared = tf.reduce_sum(tf.mul(mean_vec, mean_vec, name='l3'), reduction_indices=[1], name='l4')

    kl_div = tf.div(tf.sub(tf.add(cov_trace, mean_squared, name='l5'),
                    tf.add(k, tf.log(cov_determinant + tf.to_float(0.0000001), name='l6'), name='l7'), name='l8'),
                    tf.to_float(2), name='KL-divergence')

    print("kldiv: ", kl_div.get_shape())
    # reconstruction error
    diff = target - vae
    print("diff: ", diff.get_shape())
    rec_err = tf.reduce_prod(tf.mul(diff, diff, name='l9'), reduction_indices=[1], name='rec_err')

    # so the error happens somewhere in the computation of the rec_err gradient.
    # both the neg_lower_bound (duh) and the KL-divergence gradients come before (not l8 though) and seem fine

    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound
    bound = tf.add(kl_div, rec_err, name='neg_lower_bound')
    print("bound: ", bound.get_shape())
    return bound


def training(bound, learning_rate):

    tf.scalar_summary(bound.op.name, bound)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(bound, global_step=global_step)
    return train_op
