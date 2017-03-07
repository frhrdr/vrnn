import tensorflow as tf
import numpy as np
from vrnn_model import gaussian_kl_div, gaussian_log_p, optimization, loss


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

    return mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, bin_x, f_out, f_state


def loss_plus_binary_CE(x_target, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, bin_x, param_dict, watchlist):

    # split off last feature as binary
    x_target, bin_target = tf.split(x_target, [param_dict['data_dim'] - 1, 1])

    # compute binary CE
    ce_loss = bin_target * tf.log(bin_x) + (1 - bin_target) * tf.log(1 - bin_x)
    # take masking into account
    if param_dict['masking']:
        mask = tf.abs(bin_target - tf.constant(param_dict['mask_value']))
        # mask = tf.sign(tf.reduce_max(zero_vals, axis=1))
        num_live_samples = tf.reduce_sum(mask, axis=0)
        ce_loss *= mask
        bound = tf.reduce_sum(ce_loss, name='avg_neg_lower_bound') / num_live_samples
    else:
        # average over samples
        ce_loss = tf.reduce_mean(ce_loss, name='avg_neg_lower_bound')
    # compute and add loss for rest
    bound = loss(x_target, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict, watchlist)
    # return bound
    bound += ce_loss

    return bound
