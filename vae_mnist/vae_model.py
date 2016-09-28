import tensorflow as tf
import math


def inference(images, batch_size, encoder_fun, decoder_fun, n_latentvars, img_dims):

    # make encoder
    encoder_nn = encoder_fun(images, img_dims, 2*n_latentvars)

    # create mean and covariance (vec of cov is easier to use)
    mean_vec = tf.slice(encoder_nn, [0, 0], [batch_size, n_latentvars], name='mu')
    cov_vec = tf.slice(encoder_nn, [0, n_latentvars], [batch_size, n_latentvars], name='Sigma')

    # draw samples
    noise_vec = tf.random_normal([n_latentvars], mean=0.0, stddev=1.0, dtype=tf.float32, seed=123, name='z_noise')
    z_vec = mean_vec + tf.mul(cov_vec, noise_vec)

    # make decoder
    # if decoder_fun is None:
    #     with tf.name_scope('decoder'):
    #         decoder_nn = simple_mlp(z_vec, n_latentvars, 128, img_dims)
    # else:
    decoder_nn = decoder_fun(z_vec, n_latentvars, img_dims)
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

    # reconstruction error
    diff = target - vae

    rec_err = tf.reduce_sum(tf.mul(diff, diff, name='l9'), reduction_indices=[1], name='rec_err')

    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound
    bound = tf.add(kl_div, rec_err, name='neg_lower_bound')

    # average over samples
    bound = tf.reduce_mean(bound, name='avg_neg_lower_bound')
    return bound


def training(bound, learning_rate):

    tf.scalar_summary(bound.op.name, bound)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(bound, global_step=global_step)
    return train_op


def generation(decoder_fun, n_latentvars, img_dims):
    z_vec = tf.random_normal([100, n_latentvars])
    decoder_nn = decoder_fun(z_vec, n_latentvars, img_dims)
    return decoder_nn
