import tensorflow as tf
import numpy as np


def sample(params, noise, dist='gauss'):
    if 'gauss' in dist:
        mean, cov = params
        s = mean + tf.sqrt(cov) * noise
    elif 'gm' in dist:
        means, covs, pis = params
        eps_x, eps_pi = noise
        s = means[eps_pi] + tf.sqrt(covs[eps_pi]) * eps_x
    else:
        raise NotImplementedError
    return s


def inference(in_pl, hid_pl, f_state, eps_z, param_dict, fd):
    phi_x = fd['phi_x'](in_pl)
    mean_0, cov_0 = fd['phi_prior'](hid_pl)
    mean_z, cov_z = fd['phi_enc'](phi_x, hid_pl)
    z = sample((mean_z, cov_z), eps_z, 'gauss')
    phi_z = fd['phi_z'](z)
    params_out = fd['phi_dec'](phi_z, hid_pl)
    f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)
    return mean_0, cov_0, mean_z, cov_z, params_out, f_out, f_state


def gaussian_log_p(params_out, x_target, dim):
    mean_x, cov_x = params_out

    x_diff = x_target - mean_x
    x_square = tf.reduce_sum((x_diff / cov_x) * x_diff, axis=[1])
    log_x_exp = -0.5 * x_square
    log_cov_x_det = tf.reduce_sum(tf.log(cov_x), axis=[1])
    log_x_norm = -0.5 * (dim * tf.log(2 * np.pi) + log_cov_x_det)
    log_p = log_x_norm + log_x_exp
    return log_p, log_x_norm, log_x_exp


def gm_log_p(params_out, x_target, dim):
    mean_x, cov_x, pi_x = params_out
    mean_x = tf.transpose(mean_x, perm=[1, 0, 2])
    cov_x = tf.transpose(cov_x, perm=[1, 0, 2])
    pi_x = tf.transpose(pi_x, perm=[1, 0])

    x_diff = x_target - mean_x
    x_square = tf.reduce_sum((x_diff / cov_x) * x_diff, axis=[2])
    log_x_exp = -0.5 * x_square
    log_cov_x_det = tf.reduce_sum(tf.log(cov_x), axis=[2])
    log_x_norm = -0.5 * (dim * tf.log(2 * np.pi) + log_cov_x_det) + pi_x
    log_p = tf.reduce_logsumexp(log_x_norm + log_x_exp, axis=[0])
    return log_p, log_x_norm, log_x_exp


def gaussian_kl_div(mean_0, cov_0, mean_1, cov_1, dim):
    mean_diff = mean_1 - mean_0
    cov_1_inv = tf.reciprocal(cov_1)
    log_cov_1_det = tf.reduce_sum(tf.log(cov_1), axis=[1])
    log_cov_0_det = tf.reduce_sum(tf.log(cov_0), axis=[1])
    log_term = log_cov_1_det - log_cov_0_det
    trace_term = tf.reduce_sum(cov_1_inv * cov_0, axis=[1])
    square_term = tf.reduce_sum(mean_diff * cov_1_inv * mean_diff, axis=[1])
    kl_div = 0.5 * (trace_term + square_term - dim + log_term)
    return kl_div


def loss(x_target, mean_0, cov_0, mean_z, cov_z, params_out, param_dict):
    kl_div = gaussian_kl_div(mean_z, cov_z, mean_0, cov_0, param_dict['z_dim'])
    if param_dict['model'] == 'gauss_out':
        log_p, log_x_norm, log_x_exp = gaussian_log_p(params_out, x_target, param_dict['z_dim'])
    else:
        log_p, log_x_norm, log_x_exp = gm_log_p(params_out, x_target, param_dict['z_dim'])

    if param_dict['masking']:
        zero_vals = tf.abs(x_target - tf.constant(param_dict['mask_value'], dtype=tf.float32))
        mask = tf.sign(tf.reduce_max(zero_vals, axis=1))
        num_live_samples = tf.reduce_sum(mask, axis=0)
        log_p = tf.reduce_sum(log_p * mask, name='log_p_sum') / num_live_samples
        kl_div = tf.reduce_sum(kl_div * mask, name='kl_div_sum') / num_live_samples
        bound = kl_div - log_p
    else:
        kl_div = tf.reduce_mean(kl_div, name='kldiv_scalar')
        log_p = tf.reduce_mean(log_p, name='log_p_scalar')
        bound = kl_div - log_p

    norm = tf.reduce_mean(log_x_norm, name='norm_scalar')
    exp = tf.reduce_mean(log_x_exp, name='exp_scalar')
    return bound, kl_div, log_p, norm, exp


def optimization(err_acc, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    # grads = tf.gradients(err_acc, tvars)
    grads = [tf.clip_by_value(k, -100, 100) for k in tf.gradients(err_acc, tvars)]
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return train_op


def train_loop(x_pl, f_theta, err_acc, count, f_state, eps_z, param_dict, fun_dict, debug_tensors):
    x_t = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    eps_z_t = tf.squeeze(tf.slice(eps_z, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    mean_0, cov_0, mean_z, cov_z, params_out, f_theta, f_state = inference(x_t, f_theta, f_state, eps_z_t,
                                                                           param_dict, fun_dict)
    bound_step, kldiv_step, log_p_step, norm_step, exp_step = loss(x_t, mean_0, cov_0, mean_z, cov_z,
                                                                   params_out, param_dict)
    bound_acc = err_acc[0] + bound_step
    kldiv_acc = err_acc[1] + kldiv_step
    log_p_acc = err_acc[2] + log_p_step
    norm_acc = err_acc[3] + norm_step
    exp_acc = err_acc[4] + exp_step

    err_acc = [bound_acc, kldiv_acc, log_p_acc, norm_acc, exp_acc]
    if param_dict['model'] == 'gm_out':
        mean_x, cov_x, _ = params_out
    else:
        mean_x, cov_x = params_out

    debug_tensors = [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x]
    count += 1
    return x_pl, f_theta, err_acc, count, f_state, eps_z, debug_tensors


def get_train_loop_fun(param_dict, fun_dict):
    # function wrapper to assign the dicts. return value can be looped with tf.while_loop
    def train_loop_fun(x_pl, hid_pl, err_acc, count, f_state, eps_z, debug_tensors):
        return train_loop(x_pl, hid_pl, err_acc, count, f_state, eps_z, param_dict, fun_dict, debug_tensors)
    return train_loop_fun


def get_train_stop_fun(num_iter):
    def train_stop_fun(*args):
        count = args[3]
        return tf.less(count, num_iter)
    return train_stop_fun


def generation(hid_pl, f_state, eps_z, eps_out, pd, fd):
    params_prior = fd['phi_prior'](hid_pl)
    z = sample(params_prior, eps_z, 'gauss')
    phi_z = fd['phi_z'](z)
    params_out = fd['phi_dec'](phi_z, hid_pl)
    x = sample(params_out, eps_out, pd['model'])

    phi_x = fd['phi_x'](x)
    f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)
    return x, f_out, f_state


def gen_loop(x_pl, hid_pl, count, f_state, eps_z, eps_out, pd, fun_dict):
    if pd['model'] == 'gauss_out':
        eps_x = eps_out
    elif pd['model'] == 'gm_out':
        eps_x = eps_out[0]
    else:
        raise NotImplementedError

    eps_z_t = tf.squeeze(tf.slice(eps_z, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    eps_x_t = tf.squeeze(tf.slice(eps_x, [tf.to_int32(count), 0, 0], [1, -1, -1]))

    if pd['model'] == 'gauss_out':
        eps_out_t = eps_x_t
    elif pd['model'] == 'gm_out':
        eps_pi = eps_out[1]
        eps_pi_t = tf.squeeze(tf.slice(eps_pi, [tf.to_int32(count), 0], [1, -1]))
        eps_out_t = (eps_x_t, eps_pi_t)
    else:
        raise NotImplementedError

    x_t, f_out, f_state = generation(hid_pl, f_state, eps_z_t, eps_out_t, pd, fun_dict)

    x_old = tf.slice(x_pl, [0, 0, 0], [tf.to_int32(count), -1, -1])
    x_empty = tf.slice(x_pl, [tf.to_int32(count) + 1, 0, 0], [-1, -1, -1])
    x_t = tf.reshape(x_t, [1, pd['batch_size'], pd['x_dim']])
    x_pl = tf.concat([x_old, x_t, x_empty], axis=0)
    x_pl.set_shape([pd['seq_length'], pd['batch_size'], pd['x_dim']])

    count += 1
    return x_pl, f_out, count, f_state, eps_z, eps_out


def get_gen_loop_fun(param_dict, fun_dict):
    def f(x_pl, hid_pl, count, f_state, eps_z, eps_out):
        return gen_loop(x_pl, hid_pl, count, f_state, eps_z, eps_out, param_dict, fun_dict)
    return f


def get_gen_stop_fun(num_iter):
    def gen_stop_fun(*args):
        count = args[2]
        return tf.less(count, num_iter)
    return gen_stop_fun

    # grads = [tf.clip_by_value(k, -1000, 1000) for k in tf.gradients(err_acc, tvars)]
    # grads, _ = tf.clip_by_global_norm(tf.gradients(err_acc, tvars), 1)
    # grads = [tf.clip_by_norm(k, 1) for k in tf.gradients(err_acc, tvars)]

    # DEBUG
    # all_grads = tf.gradients(err_acc, tvars)
    # abs_grads = [tf.abs(k) for k in all_grads if k is not None]
    # max_grads = [tf.reduce_max(k) for k in abs_grads]
    # mean_grads = [tf.reduce_mean(k) for k in abs_grads]
    # grad_print = tf.Print(grads[0], max_grads, summarize=1, message='max_g ')
    # grad_print = tf.Print(grad_print, mean_grads, summarize=1, message='mean_g ')
    #
    # abs_grads = [tf.abs(k) for k in grads if k is not None]
    # max_grads = [tf.reduce_max(k) for k in abs_grads]
    # mean_grads = [tf.reduce_mean(k) for k in abs_grads]
    # grad_print = tf.Print(grad_print, max_grads, summarize=1, message='max_g_c ')
    # grad_print = tf.Print(grad_print, mean_grads, summarize=1, message='mean_g_c ')

# def naive_rec_err(mean_x, cov_x, x_target, k):  # for debugging purposes
#     x_diff = x_target - mean_x
#     return -tf.reduce_sum(x_diff * x_diff, axis=[1])


