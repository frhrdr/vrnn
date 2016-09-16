from __future__ import print_function
import tensorflow as tf
import vae_model


class DecoderNN:

    def __init__(self):
        pass

    def inference(self, z_tensor, img_dims=28**2, ckpt_file=None):
        pass
        # build NN here
        # can I load the weights without a session i might need later?
        decoder = vae_model.simple_mlp(z_tensor, 128, img_dims)

    def generator(self, z_dims, img_dims, ckpt_file):
        z = tf.random_normal([z_dims])
        return self.inference(z, img_dims, ckpt_file)
