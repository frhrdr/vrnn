PARAM_DICT = dict()

# data path
PARAM_DICT['data_path'] = 'data/series2_1000n_20d_30t.npy'
PARAM_DICT['log_path'] = 'data/logs/test1'
PARAM_DICT['log_freq'] = 10
PARAM_DICT['print_freq'] = 5

# specify global settings
PARAM_DICT['num_batches'] = 50
PARAM_DICT['batch_size'] = 20
PARAM_DICT['data_dim'] = 20
PARAM_DICT['n_latent'] = 20
PARAM_DICT['seq_length'] = 30  # flexible?
PARAM_DICT['learning_rate'] = 0.01
PARAM_DICT['max_iter'] = 20
PARAM_DICT['hid_state_size'] = 11

# infer some necessary network sizes
n_in = PARAM_DICT['data_dim']           # x
n_out = PARAM_DICT['data_dim']          # x
n_z = PARAM_DICT['n_latent']            # z
n_ms = 2 * PARAM_DICT['n_latent']       # mu + sigma
n_ht = PARAM_DICT['hid_state_size']     # h_t

# assign shared variables
phi_x_out = 10
phi_z_out = 10

# specify each net
PARAM_DICT['phi_x'] = {'name': 'phi_x',
                       'nn_type': 'simple_mlp',
                       'layers': [n_in, 10, phi_x_out]}

PARAM_DICT['phi_prior'] = {'name': 'phi_prior',
                           'nn_type': 'simple_mlp',
                           'layers': [n_ht, 10, n_ms]}

PARAM_DICT['phi_enc'] = {'name': 'phi_enc',
                         'nn_type': 'simple_mlp',
                         'layers': [phi_x_out + n_ht, 10, n_ms]}

PARAM_DICT['phi_z'] = {'name': 'phi_z',
                       'nn_type': 'simple_mlp',
                       'layers': [n_z, 10, phi_z_out]}

PARAM_DICT['phi_dec'] = {'name': 'phi_dec',
                         'nn_type': 'simple_mlp',
                         'layers': [phi_z_out + n_ht, 10, 2*n_out]}

PARAM_DICT['f_theta'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [phi_x_out + phi_z_out + n_ht, 10, n_ht]}
