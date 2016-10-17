from __future__ import print_function
import tensorflow as tf
import time
import os.path
import vae_model
import utilities as util
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from params import *

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
#                      'Must divide evenly into the dataset sizes.')
# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                      'for unit testing.')


# data_sets = input_data.read_data_sets('data', False)
# print(data_sets)
# print(data_sets.train.next_batch(5))


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMG_DIMS))
    target_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMG_DIMS))
    return images_placeholder, target_placeholder


def fill_feed_dict(data_set, images_pl, target_pl):
    images_feed, _ = data_set.next_batch(BATCH_SIZE)

    feed_dict = {
        images_pl: images_feed,
        target_pl: images_feed,
    }
    return feed_dict


def run_training(encoder_fun=None, decoder_fun=None):
    data_sets = input_data.read_data_sets(DATA_DIR)

    with tf.Graph().as_default():
        images_placeholder, target_placeholder = placeholder_inputs(BATCH_SIZE)

        if encoder_fun is None:
            encoder_fun = lambda t_in, in_dims, n_out: util.simple_mlp(t_in, in_dims, DEFAULT_N_HIDDEN, n_out, 'encoder')
        if decoder_fun is None:
            decoder_fun = lambda t_in, in_dims, n_out: util.simple_mlp(t_in, in_dims, DEFAULT_N_HIDDEN, n_out, 'decoder')

        vae, mean_vec, cov_vec = vae_model.inference(images_placeholder, BATCH_SIZE,
                                                     encoder_fun, decoder_fun, n_latentvars=N_LATENT, img_dims=IMG_DIMS)
        bound = vae_model.loss(vae, mean_vec, cov_vec, target_placeholder, n_latentvars=N_LATENT)
        train_op = vae_model.training(bound, learning_rate=LEARNING_RATE)

        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter('data', sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in xrange(MAX_STEPS):

            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       target_placeholder)

            _, loss_value = sess.run([train_op, bound],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:

                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(LOG_DIR, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=(step + 1))

            # maybe later add validation and test eval


def run_generation(ckpt_file, decoder_fun, n_latentvars, img_dims, n_samples):

    # rebuild decoder with saved weights
    with tf.Graph().as_default():
        generator = vae_model.generation(decoder_fun, n_latentvars, img_dims)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            # sample z values and plot them assuming images are square (for now)
            img_mat = np.zeros((n_samples, img_dims))
            for idx in range(n_samples):
                img_mat[idx, :] = sess.run(generator)[0, :]

            n = int(np.sqrt(img_dims))
            assert n**2 == img_dims
            img_mat = img_mat.reshape((n_samples, n, n))
            util.plot_img_mats(img_mat)
