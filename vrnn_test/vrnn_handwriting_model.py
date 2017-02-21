import tensorflow as tf
import numpy as np
from vrnn_model import gaussian_kl_div, gaussian_log_p, train


def inference_plus_binary(in_pl, hid_pl, f_state, eps_z, param_dict, fun_dict, watchlist):

    # rename for brevity
    pd = param_dict
    fd = fun_dict

    phi_x = fd['phi_x'](in_pl)
    mean_0, cov_0 = fd['phi_prior'](hid_pl)
    mean_z, cov_z = fd['phi_enc'](phi_x, hid_pl)
    z = mean_z + tf.sqrt(cov_z) * eps_z
    phi_z = fd['phi_z'](z)
    mean_x, cov_x, bin_x = fd['phi_dec'](phi_z, hid_pl)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)

    # DEBUG
    # f_out = tf.Print(f_out, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x], message="mc_0zx ")

    return mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, f_out, f_state


def loss_plus_binary_CE(x_target, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict, watchlist):

    # split off last feature as binary
    gauss, binary = tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1])

    # compute binary CE

    # take masking into account

    # compute and add loss for rest

    # return bound
    return bound
