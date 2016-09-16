from __future__ import print_function
import tensorflow as tf
import time
import os.path
import vae_model
from tensorflow.examples.tutorials.mnist import input_data

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

max_steps = 505
train_dir = 'data'
batch_size = 100
n_latent = 30
# data_sets = input_data.read_data_sets('data', False)
# print(data_sets)
# print(data_sets.train.next_batch(5))


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28**2))
    target_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28**2))
    return images_placeholder, target_placeholder


def fill_feed_dict(data_set, images_pl, target_pl):
    images_feed, _ = data_set.next_batch(batch_size)

    feed_dict = {
        images_pl: images_feed,
        target_pl: images_feed,
    }
    return feed_dict


def run_training():
    data_sets = input_data.read_data_sets(train_dir)

    with tf.Graph().as_default():
        images_placeholder, target_placeholder = placeholder_inputs(batch_size)

        vae, mean_vec, cov_vec = vae_model.inference(images_placeholder, batch_size, n_latentvars=n_latent)
        bound = vae_model.loss(vae, mean_vec, cov_vec, target_placeholder, n_latentvars=n_latent)
        train_op = vae_model.training(bound, learning_rate=0.1)

        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter('data', sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in xrange(max_steps):

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

            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join(train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

            # maybe later add validation and test eval

run_training()
