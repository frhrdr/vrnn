import tensorflow as tf
import numpy as np


def vanilla_inference(in_pl, hid_pl, f_state, eps_z, param_dict, fun_dict, watchlist):

    # rename for brevity
    pd = param_dict
    fd = fun_dict

    phi_x = fd['phi_x'](in_pl)
    mean_0, cov_0 = fd['phi_prior'](hid_pl)
    mean_z, cov_z = fd['phi_enc'](phi_x, hid_pl)
    z = mean_z + tf.sqrt(cov_z) * eps_z
    phi_z = fd['phi_z'](z)
    mean_x, cov_x = fd['phi_dec'](phi_z, hid_pl)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)

    # DEBUG
    # f_out = tf.Print(f_out, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x], message="mc_0zx ")

    return mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, f_out, f_state


def naive_rec_err(mean_x, cov_x, x_target, k):
    x_diff = x_target - mean_x
    return -tf.reduce_sum(x_diff * x_diff, axis=[1])


def gaussian_log_p(mean_x, cov_x, x_target, k):
    x_diff = x_target - mean_x
    x_square = tf.reduce_sum((x_diff / cov_x) * x_diff, axis=[1])
    log_x_exp = -0.5 * x_square
    log_cov_x_det = tf.reduce_sum(tf.log(cov_x), axis=[1])
    log_x_norm = -0.5 * (k * tf.log(2*np.pi) + log_cov_x_det)
    log_p = log_x_norm + log_x_exp
    # DEBUG
    # log_p = tf.Print(log_p, [tf.reduce_max(log_p)], message='log p ')
    # log_p = tf.Print(log_p, [log_x_exp, log_x_norm, x_square, cov_x, x_diff], message='log p comps ')
    return log_p


def gaussian_kl_div(mean_0, cov_0, mean_1, cov_1, k):
    mean_diff = mean_1 - mean_0
    cov_1_inv = tf.reciprocal(cov_1)
    log_cov_1_det = tf.reduce_sum(tf.log(cov_1), axis=[1])
    log_cov_0_det = tf.reduce_sum(tf.log(cov_0), axis=[1])

    log_term = log_cov_1_det - log_cov_0_det
    trace_term = tf.reduce_sum(cov_1_inv * cov_0, axis=[1])
    square_term = tf.reduce_sum(mean_diff * cov_1_inv * mean_diff, axis=[1])

    kl_div = 0.5 * (trace_term + square_term - k + log_term)
    # DEBUG
    # kl_div = tf.Print(kl_div, [tf.reduce_min(kl_div)], message="kl_div ")
    # kl_div = tf.Print(kl_div, [trace_term, square_term, log_term], message="kl-comps ")
    return kl_div


def vanilla_loss(x_target, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict, watchlist):

    k = param_dict['n_latent']

    # log_p = gaussian_log_p(mean_x, cov_x, x_target, k)
    log_p = naive_rec_err(mean_x, cov_x, x_target, k)
    kl_div = log_p
    # kl_div = gaussian_kl_div(mean_z, cov_z, mean_0, cov_0, k)

    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound

    # DEBUG
    # bound = tf.Print(bound, [bound], message='bound ')

    if param_dict['masking']:
        zero_vals = tf.abs(x_target - tf.constant(param_dict['mask_value'], dtype=tf.float32))
        mask = tf.sign(tf.reduce_max(zero_vals, axis=1))
        num_live_samples = tf.reduce_sum(mask, axis=0)

        # new - passing kldiv and log_p for plotting outside loop
        log_p = tf.reduce_sum(log_p * mask, name='log_p_sum') / num_live_samples
        kl_div = tf.reduce_sum(kl_div * mask, name='kl_div_sum') / num_live_samples
        bound = kl_div - log_p
        # old
        # bound = (kl_div - log_p) * mask
        # bound = tf.reduce_sum(bound, name='avg_neg_lower_bound') / num_live_samples

    else:
        # new
        kl_div = tf.reduce_mean(kl_div, name='kldiv_scalar')
        log_p = tf.reduce_mean(log_p, name='log_p_scalar')
        bound = kl_div - log_p
        # old
        # bound = kl_div - log_p
        # # average over samples
        # bound = tf.reduce_mean(bound, name='avg_neg_lower_bound')

    # DEBUG
    # grads = [tf.reduce_mean(k) for k in tf.gradients(bound, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x])]
    # grads = [tf.reduce_mean(k) for k in tf.gradients(square_term, [mean_z])]
    # grads.append(tf.reduce_mean(cov_0_inv))
    # bound = tf.Print(bound, grads, message='bg ')
    # bound = tf.Print(bound, tf.gradients(bound, [mean_z, cov_z]), message='bg ')

    return bound, kl_div, log_p


def train(err_acc, learning_rate):
    tf.summary.scalar(err_acc.op.name, err_acc)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()
    grads = tf.gradients(err_acc, tvars)
    # grads = [tf.clip_by_value(k, -1000, 1000) for k in tf.gradients(err_acc, tvars)]
    # grads, _ = tf.clip_by_global_norm(tf.gradients(err_acc, tvars), 1)
    # grads = [tf.clip_by_norm(k, 1) for k in tf.gradients(err_acc, tvars)]


    # DEBUG
    all_grads = tf.gradients(err_acc, tvars)
    abs_grads = [tf.abs(k) for k in all_grads if k is not None]
    max_grads = [tf.reduce_max(k) for k in abs_grads]
    mean_grads = [tf.reduce_mean(k) for k in abs_grads]
    grad_print = tf.Print(grads[0], max_grads, summarize=1, message='max_g ')
    grad_print = tf.Print(grad_print, mean_grads, summarize=1, message='mean_g ')

    abs_grads = [tf.abs(k) for k in grads if k is not None]
    max_grads = [tf.reduce_max(k) for k in abs_grads]
    mean_grads = [tf.reduce_mean(k) for k in abs_grads]
    grad_print = tf.Print(grad_print, max_grads, summarize=1, message='max_g_c ')
    grad_print = tf.Print(grad_print, mean_grads, summarize=1, message='mean_g_c ')
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    # print([(k.name, k.get_shape()) for k in tvars])
    # train_op = optimizer.minimize(err_acc)  # , global_step=global_step)

    return train_op, grad_print


def train_loop(x_pl, hid_pl, err_acc, count, f_state, eps_z, param_dict, fun_dict, watchlist):
    # the dicts must be assigned before looping over a parameter subset (lambda) of this function

    # set x_pl to elem of list indexed by count
    x_t = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    eps_z_t = tf.squeeze(tf.slice(eps_z, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    # build inference model
    mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, f_theta, f_state = vanilla_inference(x_t, hid_pl, f_state, eps_z_t,
                                                                                      param_dict, fun_dict, watchlist)
    # build loss
    bound_step, kldiv_step, log_p_step = vanilla_loss(x_t, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict, watchlist)
    # set hid_pl to result of f_theta
    hid_pl = f_theta
    # set err_acc to += error from this time-step
    # err_acc = tf.Print(err_acc, [err_acc], message="err_acc_loop")

    bound_acc = err_acc[0] + bound_step
    kldiv_acc = err_acc[1] + kldiv_step
    log_p_acc = err_acc[2] + log_p_step
    err_acc = [bound_acc, kldiv_acc, log_p_acc]

    count = count + 1
    return x_pl, hid_pl, err_acc, count, f_state, eps_z


def get_train_loop_fun(param_dict, fun_dict, watchlist):
    # function wrapper to assign the dicts. return value can be looped with tf.while_loop
    def train_loop_fun(x_pl, hid_pl, err_acc, count, f_state, eps_z):
        return train_loop(x_pl, hid_pl, err_acc, count, f_state, eps_z, param_dict, fun_dict, watchlist)
    return train_loop_fun


def get_train_stop_fun(num_iter):
    def train_stop_fun(*args):
        count = args[3]
        return tf.less(count, num_iter)
    return train_stop_fun












def generation(hid_pl, f_state, eps_z, eps_x, param_dict, fun_dict):
    # rename for brevity
    pd = param_dict
    fd = fun_dict

    mean_0, cov_0 = fd['phi_prior'](hid_pl)
    z = mean_0 + tf.sqrt(cov_0) * eps_z
    phi_z = fd['phi_z'](z)
    mean_x, cov_x = fd['phi_dec'](phi_z, hid_pl)
    x = mean_x + tf.sqrt(cov_x) * eps_x
    phi_x = fd['phi_x'](x)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)
    # f_out = tf.Print(f_out, [mean_0, cov_0, mean_x, cov_x], message='mc_0x ')
    # f_out = fd['f_theta'](f_in)
    # f_out = tf.Print(f_out, f_state, message="f_state")
    # f_out = tf.Print(f_out, [f_out, f_in], message="f_out", summarize=10)
    return x, f_out, f_state


def gen_loop(x_pl, hid_pl, count, f_state, eps_z, eps_x, param_dict, fun_dict):
    pd = param_dict

    eps_z_t = tf.squeeze(tf.slice(eps_z, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    eps_x_t = tf.squeeze(tf.slice(eps_x, [tf.to_int32(count), 0, 0], [1, -1, -1]))

    x_t, f_out, f_state = generation(hid_pl, f_state, eps_z_t, eps_x_t, param_dict, fun_dict)

    x_old = tf.slice(x_pl, [0, 0, 0], [tf.to_int32(count), -1, -1])
    x_empty = tf.slice(x_pl, [tf.to_int32(count) + 1, 0, 0], [-1, -1, -1])
    x_t = tf.reshape(x_t, [1, pd['batch_size'], pd['data_dim']])
    x_pl = tf.concat([x_old, x_t, x_empty], axis=0)
    x_pl.set_shape([pd['seq_length'], pd['batch_size'], pd['data_dim']])

    count = tf.add(count, 1)

    return x_pl, f_out, count, f_state, eps_z, eps_x


def get_gen_loop_fun(param_dict, fun_dict):
    def f(x_pl, hid_pl, count, f_state, eps_z, eps_x):
        return gen_loop(x_pl, hid_pl, count, f_state, eps_z, eps_x, param_dict, fun_dict)
    return f


def get_gen_stop_fun(num_iter):
    def gen_stop_fun(*args):
        count = args[2]
        return tf.less(count, num_iter)
    return gen_stop_fun





















# def latent_gm_inference(in_pl, hid_pl, f_state, eps_z, eps_pi_z, param_dict, fun_dict, watchlist):
#
#     # rename for brevity
#     pd = param_dict
#     fd = fun_dict
#
#     phi_x = fd['phi_x'](in_pl)
#     means_0, covs_0, pis_0 = fd['phi_prior'](hid_pl)
#     means_z, covs_z, pis_z = fd['phi_enc'](phi_x, hid_pl)
#     z = mean_z + tf.sqrt(cov_z) * eps_z
#     pi_sums = tf.accumulate_n(pis_z)
#     for mean, cov, pi, pi_sum in zip(means_z, covs_z, pis_z, pi_sums):
#         pass
#     phi_z = fd['phi_z'](z)
#     mean_x, cov_x = fd['phi_dec'](phi_z, hid_pl)
#
#     # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
#     f_in = tf.concat([phi_x, phi_z], axis=1, name='f_theta_joint_inputs')
#     f_out, f_state = fd['f_theta'](f_in, f_state)
#
#     # DEBUG
#     # f_out = tf.Print(f_out, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x], message="mc_0zx ")
#     raise NotImplementedError
#     return means_0, covs_0, pis_0, means_z, covs_z, pis_0, mean_x, cov_x, f_out, f_state
#
#
# def gaussian_mixture_kl_div_bound(means_0, covs_0, pis_0, means_z, covs_z, pis_z, k, param_dict):
#     # with matched bound approximation (13) in J Hershey, P Olsen:
#     # "Approximating the Kullback Leibler Divergence Between Gaussian Mixture Models"
#     kl_acc = tf.constant(0, dtype=tf.float32, shape=(param_dict['batch_size'],))
#
#     for mean_a, cov_a, pi_a, mean_b, cov_b, pi_b in zip(means_0, covs_0, pis_0, means_z, covs_z, pis_z):
#         kl_acc += pi_a * (tf.log(pi_a / pi_b) + gaussian_kl_div(mean_a, cov_a, mean_b, cov_b, k))
#
#     return kl_acc
#
#
# def gaussian_mixture_log_p():
#     raise NotImplementedError
#
