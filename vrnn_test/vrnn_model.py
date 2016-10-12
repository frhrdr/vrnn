import tensorflow as tf



def inference(in_pl, hid_pl, f_state, param_dict, fun_dict):

    # rename for brevity
    pd = param_dict
    fd = fun_dict

    phi_x = fd['phi_x'](in_pl)

    phi_enc = fd['phi_enc'](phi_x, hid_pl)
    mean_x = tf.slice(phi_enc, [0, 0], [pd['batch_size'], pd['n_latent']], name='mean_x')
    cov_x = tf.slice(phi_enc, [0, pd['n_latent']], [pd['batch_size'], pd['n_latent']], name='cov_x')

    phi_prior = fd['phi_prior'](hid_pl)
    mean_0 = tf.slice(phi_prior, [0, 0], [pd['batch_size'], pd['n_latent']], name='mean_0')
    cov_0 = tf.slice(phi_prior, [0, pd['n_latent']], [pd['batch_size'], pd['n_latent']], name='cov_0')

    # make sure covariances are not 0
    cov_0 = tf.add(cov_0, tf.to_float(0.01))
    cov_x = tf.add(cov_x, tf.to_float(0.01))

    noise = tf.random_normal([pd['n_latent']], mean=0.0, stddev=1.0, dtype=tf.float32, seed=123, name='z_noise')
    z = tf.add(mean_x, tf.mul(cov_x, noise), name='z_vec')

    phi_z = fd['phi_z'](z)
    phi_dec = fd['phi_dec'](phi_z, hid_pl)

    # f_theta being an rnn must be handled differently (maybe this inconsistency can be fixed later on)
    f_in = tf.concat(1, [hid_pl, phi_x, phi_z], name="f_theta_joint_inputs")
    # f_out, f_state = fd['f_theta'](f_in, f_state)
    f_out = fd['f_theta'](f_in)
    # f_out = tf.Print(f_out, f_state, message="f_state")
    # f_out = tf.Print(f_out, [f_out, f_in], message="f_out", summarize=10)
    return phi_dec, mean_0, cov_0, mean_x, cov_x, f_out, f_state


def loss(target, phi_dec, mean_0, cov_0, mean_x, cov_x, param_dict):
    # make tensor for reconstruction err from target and phi_dec
    diff = tf.sub(target, phi_dec, name='re1')
    # diff = tf.Print(diff, [diff], message="diff")
    rec_err = tf.reduce_sum(tf.mul(diff, diff, name='re2'), reduction_indices=[1], name='rec_err')
    # rec_err = tf.Print(rec_err, [rec_err], message="rec_err")
    # make tensor for KL divergence from means and covariance vectors
    # following equation 6 from the VAE tutorial
    k = tf.to_float(param_dict['n_latent'])
    mean_diff = tf.sub(mean_0, mean_x, name='kl1')
    cov_0_inv = tf.inv(cov_0)
    cov_x = tf.Print(cov_x, [cov_x, mean_x], message="cov_and_mean_x")
    cov_0_det = tf.reduce_prod(cov_0, reduction_indices=[1], name='kl2')
    cov_x_det = tf.reduce_prod(cov_x, reduction_indices=[1], name='kl3')
    trace_term = tf.reduce_sum(tf.mul(cov_0_inv, cov_x), reduction_indices=[1], name='kl4')
    square_term = tf.reduce_sum(tf.mul(tf.mul(mean_diff, cov_0_inv, name='kl5'),
                                       mean_diff, name='kl6'),
                                reduction_indices=[1], name='kl7')

    cov_x_det2 = tf.mul(cov_x_det, tf.to_float(10.0**20))  # quick fix: det^2 in derivative underflows to 0 otherwise
    cov_0_det2 = tf.mul(cov_0_det, tf.to_float(10.0**20))

    temp = tf.div(cov_0_det2, cov_x_det2, name='kl8')
    log_term = tf.log(temp, name='kl9')
    # log_term = tf.Print(log_term, [log_term], message="log_term")
    kl_div = tf.div(tf.add(tf.add(trace_term, square_term, name='kl10'),
                           tf.sub(log_term, k, name='kl11'), name='kl12'),
                    tf.to_float(2), name='kl_div')
    # kl_div = tf.Print(kl_div, [kl_div], message="kl_div")
    # kl_div = tf.Print(kl_div, [tf.gradients(temp, [cov_x_det2]), temp, cov_0_det, cov_x_det2], message="grad ")
    # negative variational lower bound
    # (optimizer can only minimize - same as maximizing positive lower bound
    bound = tf.add(rec_err, kl_div, name='neg_lower_bound')
    # average over samples
    bound = tf.reduce_mean(bound, name='avg_neg_lower_bound')

    return bound


def loop(x_pl, hid_pl, err_acc, count, f_state, param_dict, fun_dict):
    # the dicts must be assigned before looping over a parameter subset (lambda) of this function

    # set x_pl to elem of list indexed by count
    # x_t = tf.squeeze(tf.slice(x_pl, [tf.to_int32(count), 0, 0], [1, -1, -1]))
    x_t = tf.squeeze(tf.slice(x_pl, [tf.to_int32(10), 0, 0], [1, -1, -1]))
    # build inference model
    phi_dec, mean_0, cov_0, mean_x, cov_x, f_theta, f_state = inference(x_t, hid_pl, f_state, param_dict, fun_dict)
    # build loss
    step_error = loss(x_t, phi_dec, mean_0, cov_0, mean_x, cov_x, param_dict)
    # set hid_pl to result of f_theta
    hid_pl = f_theta
    # set err_acc to += error from this time-step
    # err_acc = tf.Print(err_acc, [err_acc], message="err_acc_loop")
    err_acc = tf.add(err_acc, step_error)
    # set count += 1
    count = tf.add(count, 1)
    count = tf.Print(count, [hid_pl], message="hid", summarize=20)
    return x_pl, hid_pl, err_acc, count, f_state


def get_loop_fun(param_dict, fun_dict):
    # function wrapper to assign the dicts. return value can be looped with tf.while_loop
    def loop_fun(x_list, hid_pl, err_acc, count, f_state):
        return loop(x_list, hid_pl, err_acc, count, f_state, param_dict, fun_dict)
    return loop_fun


def get_stop_fun(num_iter):
    def stop_fun(a, b, c, count, d):
        return tf.less(count, num_iter)
    return stop_fun


def train(err_acc, learning_rate):
    tf.scalar_summary(err_acc.op.name, err_acc)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    err = tf.Print(err_acc, [err_acc], message="err_acc_train")
    train_op = optimizer.minimize(err)  # , global_step=global_step)
    return train_op


def generate():
    pass
