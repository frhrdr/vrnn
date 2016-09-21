import tensorflow as tf
import math


def inference(in_pl, hid_pl, param_dict, fun_dict):

    pd = param_dict
    fd = fun_dict

    phi_x = fd['phi_x'](in_pl)

    phi_enc = fd['phi_enc'](phi_x, hid_pl)
    mean_x = tf.slice(phi_enc, [0, 0], [pd['batch_size'], pd['n_latent']], name='mean_x')
    cov_x = tf.slice(phi_enc, [0, pd['n_latent']], [pd['batch_size'], pd['n_latent']], name='cov_x')

    phi_prior = fd['phi_prior'](hid_pl)
    mean_0 = tf.slice(phi_prior, [0, 0], [pd['batch_size'], pd['n_latent']], name='mean_0')
    cov_0 = tf.slice(phi_prior, [0, pd['n_latent']], [pd['batch_size'], pd['n_latent']], name='cov_0')

    noise = tf.random_normal([pd['n_latent']], mean=0.0, stddev=1.0, dtype=tf.float32, seed=123, name='z_noise')
    z = tf.add(mean_x, tf.mul(cov_x, noise), name='z_vec')

    phi_z = fd['phi_z'](z)
    phi_dec = fd['phi_dec'](phi_z, hid_pl)

    f_theta = fd['f_theta'](hid_pl, phi_x, phi_z)


def loss():
    pass


def train():
    pass


def generate():
    pass