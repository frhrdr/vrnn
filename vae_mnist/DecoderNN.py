from __future__ import print_function
import tensorflow as tf


class DecoderNN:

    def __init__(self):
        pass

    def inference(self, z_tensor, ckpt_file=None):
        pass
        # build NN here
        # can I load the weights without a session i might need later?


    def generator(self, z_dims, ckpt_file):
        z = tf.random_normal([z_dims])
        return self.inference(z, ckpt_file)
