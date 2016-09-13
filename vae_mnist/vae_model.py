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


def inference(images, encoder_nn=None, decoder_nn=None, n_latentvars=10):

    # make encoder if not provided
    if encoder_nn is None:
        with tf.name_scope('encoder'):
            encoder_nn = simple_mlp(images, 28**2, 128, 2*n_latentvars)

    # create mean and covariance
    mean_vec = tf.slice(encoder_nn, [0], [n_latentvars], name='mu')
    cov_vec = tf.slice(encoder_nn, [n_latentvars], [n_latentvars])
    cov_mat = tf.batch_matrix_diag(cov_vec, name='Sigma')

    # draw samples
    noise_vec = tf.random_normal(n_latentvars, mean=0.0, stddev=1.0, dtype=tf.float32, seed=123, name='z_noise')
    z_vec = mean_vec + tf.mul(cov_vec, noise_vec)

    # make decoder if not provided
    if decoder_nn is None:
        with tf.name_scope('encoder'):
            decoder_nn = simple_mlp(z_vec, n_latentvars, 128, 28**2)

    return decoder_nn, mean_vec, cov_mat

def bound(vae, mean_vec, cov_mat, target):

    # KL divergence (see VAE Tut formula 7)
    # trace and determinant could be optimized.
    k = tf.to_double(mean_vec.shape[0])
    kl_div = tf.div(tf.trace(cov_mat) + tf.matmul(mean_vec, mean_vec, transpose_a=True) -
                    k + tf.log(tf.det(cov_mat)), 2, name='KL-divergence')

    # reconstruction error
    diff = target - vae
    rec_err = tf.matmul(diff, diff, transpose_a=True, name='reconstruction error')


    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound
    bound = tf.add(kl_div, rec_err, name='neg lower bound')
    return bound

def training(bound, learning_rate):

    tf.scalar_summary(bound.op.name, bound)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(bound, global_step=global_step)
    return train_op
