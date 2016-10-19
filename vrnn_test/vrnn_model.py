import tensorflow as tf
import numpy as np


def inference(in_pl, hid_pl, f_state, eps_z, param_dict, fun_dict):

    # rename for brevity
    pd = param_dict
    fd = fun_dict

    phi_x = fd['phi_x'](in_pl)
    mean_0, cov_0 = fd['phi_prior'](hid_pl)
    mean_z, cov_z = fd['phi_enc'](phi_x, hid_pl)
    z = tf.add(mean_z, tf.mul(cov_z, eps_z), name='z_vec')
    phi_z = fd['phi_z'](z)
    mean_x, cov_x = fd['phi_dec'](phi_z, hid_pl)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat(1, [hid_pl, phi_x, phi_z], name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)

    # f_out = tf.Print(f_out, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x], message="mc_0zx ")

    return mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, f_out, f_state


def loss(x_target, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict):

    k = param_dict['n_latent']
    x_diff = tf.sub(x_target, mean_x)
    x_square = tf.matmul(tf.div(x_diff, cov_x), x_diff, transpose_a=True)
    log_x_exp = -0.5 * x_square
    cov_x_det = tf.reduce_prod(cov_x, reduction_indices=[1])
    log_x_norm = -0.5 * (k * tf.log(2*np.pi) + tf.log(cov_x_det))
    log_p = tf.add(log_x_norm, log_x_exp)
    # log_p = tf.mul(x_norm, x_exp)  # chung doesn't take log here - maybe for stability
    # log_p = tf.Print(log_p, [tf.reduce_mean(tf.gradients(log_p, [cov_x]))], message='glog')
    # log_p = tf.Print(log_p, [tf.reduce_mean(tf.gradients(cov_x_det, [cov_x]))], message='gdet')
    # log_p = tf.Print(log_p, [tf.reduce_mean(tf.gradients(log_x_exp, [cov_x]))], message='gexp')
    # log_p = tf.Print(log_p, [tf.reduce_mean(tf.gradients(log_x_norm, [cov_x]))], message='gnorm')
    # log_p = tf.Print(log_p, [log_x_norm, log_x_exp, cov_x_det], message='norm_exp_sqrtdet')
    log_p = tf.Print(log_p, [log_p], message='log p ')
    log_p = tf.Print(log_p, [log_x_exp, log_x_norm], message='log p comps ')
    # make tensor for KL divergence from means and covariance vectors
    # following equation 6 from the VAE tutorial
    k = tf.to_float(k)
    mean_diff = tf.sub(mean_0, mean_z, name='kl1')
    cov_0_inv = tf.inv(cov_0)
    # cov_z = tf.Print(cov_z, [cov_z, mean_z], message="cov_and_mean_x")
    cov_0_det = tf.reduce_prod(cov_0, reduction_indices=[1], name='kl2')
    cov_z_det = tf.reduce_prod(cov_z, reduction_indices=[1], name='kl3')
    trace_term = tf.reduce_sum(tf.mul(cov_0_inv, cov_z), reduction_indices=[1], name='kl4')
    # trace_term = tf.Print(trace_term, [mean_diff, cov_0_inv, mean_0, mean_z], message='md_ci_m0_mz ')
    square_term = tf.reduce_sum(tf.mul(tf.mul(mean_diff, cov_0_inv, name='kl5'),
                                       mean_diff, name='kl6'),
                                reduction_indices=[1], name='kl7')

    # cov_z_det = tf.mul(cov_z_det, tf.to_float(10.0**20))  # quick fix: det^2 in derivative under-flows to 0 otherwise
    # cov_0_det2 = tf.mul(cov_0_det, tf.to_float(10.0**20))
    #
    temp = tf.div(cov_0_det, cov_z_det, name='kl8')
    # log_term = tf.log(temp, name='kl9')
    log_term = tf.log(cov_0_det) - tf.log(cov_z_det)
    kl_div = tf.div(tf.add(tf.add(trace_term, square_term, name='kl10'),
                           tf.sub(log_term, k, name='kl11'), name='kl12'),
                    tf.to_float(2), name='kl_div')
    kl_div = tf.Print(kl_div, [kl_div], message="kl_div ")
    # kl_div = tf.Print(kl_div, [tf.gradients(log_term, [cov_z_det]), temp, tf.log(cov_0_det), tf.log(cov_z_det)], message="grad ")
    kl_div = tf.Print(kl_div, [trace_term, square_term, log_term], message="kl-comps ")
    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound
    bound = tf.sub(kl_div, log_p, name='neg_lower_bound')
    # bound = tf.Print(bound, [bound], message='bound ')
    # average over samples
    bound = tf.reduce_mean(bound, name='avg_neg_lower_bound')
    # grads = [tf.reduce_mean(k) for k in tf.gradients(bound, [mean_0, cov_0, mean_z, cov_z, mean_x, cov_x])]
    # grads = [tf.reduce_mean(k) for k in tf.gradients(square_term, [mean_z])]
    # grads.append(tf.reduce_mean(cov_0_inv))
    # bound = tf.Print(bound, grads, message='bg ')
    # bound = tf.Print(bound, tf.gradients(bound, [mean_z, cov_z]), message='bg ')

    return bound


def train(err_acc, learning_rate):
    tf.scalar_summary(err_acc.op.name, err_acc)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()

    # grads = [tf.clip_by_value(k, -0.01, 0.01) for k in tf.gradients(err_acc, tvars)]
    grads, _ = tf.clip_by_global_norm(tf.gradients(err_acc, tvars), 1)

    # DEBUG
    abs_grads = [tf.abs(k) for k in tf.gradients(err_acc, tvars) if k is not None]
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

    # train_op = optimizer.minimize(err_acc)  # , global_step=global_step)
    return train_op, grad_print


def train_loop(x_pl, hid_pl, err_acc, count, f_state, eps_z, param_dict, fun_dict):
    # the dicts must be assigned before looping over a parameter subset (lambda) of this function

    # set x_pl to elem of list indexed by count
    x_t = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    eps_z_t = tf.squeeze(tf.slice(eps_z, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    # build inference model
    mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, f_theta, f_state = inference(x_t, hid_pl, f_state, eps_z_t,
                                                                              param_dict, fun_dict)
    # build loss
    step_error = loss(x_t, mean_0, cov_0, mean_z, cov_z, mean_x, cov_x, param_dict)
    # set hid_pl to result of f_theta
    hid_pl = f_theta
    # set err_acc to += error from this time-step
    # err_acc = tf.Print(err_acc, [err_acc], message="err_acc_loop")
    err_acc = tf.add(err_acc, step_error)
    # set count += 1
    count = tf.add(count, 1)
    # err_acc = tf.Print(err_acc, [err_acc], message='acc ')
    # count = tf.Print(count, [hid_pl], message="hid", summarize=20)
    return x_pl, hid_pl, err_acc, count, f_state, eps_z


def get_train_loop_fun(param_dict, fun_dict):
    # function wrapper to assign the dicts. return value can be looped with tf.while_loop
    def train_loop_fun(x_pl, hid_pl, err_acc, count, f_state, eps_z):
        return train_loop(x_pl, hid_pl, err_acc, count, f_state, eps_z, param_dict, fun_dict)
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
    z = tf.add(mean_0, tf.mul(cov_0, eps_z), name='z_vec')
    phi_z = fd['phi_z'](z)
    mean_x, cov_x = fd['phi_dec'](phi_z, hid_pl)
    x = tf.add(mean_x, tf.mul(cov_x, eps_x), name='z_vec')
    phi_x = fd['phi_x'](x)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat(1, [hid_pl, phi_x, phi_z], name='f_theta_joint_inputs')
    f_out, f_state = fd['f_theta'](f_in, f_state)
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
    x_pl = tf.concat(0, [x_old, x_t, x_empty])
    x_pl.set_shape([pd['seq_length'], pd['batch_size'], pd['data_dim']])
    # x_pl = tf.Print(x_pl, [tf.squeeze(tf.slice(x_pl, [0, 0, 0], [-1, 1, 1]))],
    #                 message='x ', summarize=pd['seq_length'])

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
