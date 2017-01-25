PARAM_DICT = dict()

PARAM_DICT['watchlist'] = {'allmc': []}

# data path
PARAM_DICT['series'] = 7
PARAM_DICT['data_path'] = 'data/series7_2000n_15t_4d.npy'
PARAM_DICT['log_path'] = 'data/logs/test2'
PARAM_DICT['log_freq'] = 500
PARAM_DICT['print_freq'] = 25

# settle architecture: vanilla, gm_out, gm_latent, multinomial_out, gm_latent_multinomial_out
PARAM_DICT['model'] = 'vanilla'
PARAM_DICT['split_latent'] = 1
PARAM_DICT['split_out'] = 1

# specify global settings
PARAM_DICT['num_batches'] = 50
PARAM_DICT['batch_size'] = 40
PARAM_DICT['data_dim'] = 4
PARAM_DICT['n_latent'] = 32
PARAM_DICT['seq_length'] = 15
PARAM_DICT['learning_rate'] = 0.01
PARAM_DICT['max_iter'] = 500
PARAM_DICT['hid_state_size'] = 31

# infer some necessary network sizes
n_in = PARAM_DICT['data_dim']           # x
n_out = PARAM_DICT['data_dim']          # x
n_z = PARAM_DICT['n_latent']            # z
n_ht = PARAM_DICT['hid_state_size']     # h_t

# infer number of parameters based un latent and output distribution
g_val = 2 * n_z
gm_val = (2 * n_z + 1) * PARAM_DICT['split_latent']
latent_switch = {'vanilla': g_val, 'gm_out': g_val, 'multinomial_out': g_val,
                 'gm_latent': gm_val, 'gm_latent_multinomial_out': gm_val}
n_latent_stat = latent_switch[PARAM_DICT['model']]       # mu + sigma

g_val = 2 * n_out
gm_val = (2 * n_z + 1) * PARAM_DICT['split_latent']
out_switch = {'vanilla': g_val, 'gm_out': gm_val, 'multinomial_out': n_out,
              'gm_latent': g_val, 'gm_latent_multinomial_out': n_out}
n_out_stat = out_switch[PARAM_DICT['model']]

# assign shared variables
phi_x_out = 10
phi_z_out = 10

# specify each net
PARAM_DICT['phi_x'] = {'name': 'phi_x',
                       'nn_type': 'general_mlp',
                       'activation': 'elu',
                       'layers': [n_in, 50, phi_x_out]}

PARAM_DICT['phi_prior'] = {'name': 'phi_prior',
                           'nn_type': 'general_mlp',
                           'activation': 'elu',
                           'layers': [n_ht, 50, 50, n_latent_stat],
                           'out2normal': True,
                           'init_bias': 0.0,
                           'use_batch_norm': False,
                           'splits': PARAM_DICT['split_latent']
                           }

PARAM_DICT['phi_enc'] = {'name': 'phi_enc',
                         'nn_type': 'general_mlp',
                         'activation': 'elu',
                         'layers': [phi_x_out + n_ht, 50, 50, n_latent_stat],
                         'out2normal': True,
                         'init_bias': 0.0,
                         'splits': PARAM_DICT['split_latent']
                         }

PARAM_DICT['phi_z'] = {'name': 'phi_z',
                       'nn_type': 'general_mlp',
                       'activation': 'elu',
                       'layers': [n_z, 50, phi_z_out]}

PARAM_DICT['phi_dec'] = {'name': 'phi_dec',
                         'nn_type': 'general_mlp',
                         'activation': 'elu',
                         'layers': [phi_z_out + n_ht, 50, 50, n_out_stat],
                         'out2normal': True,
                         'init_bias': 0.0,
                         'splits': PARAM_DICT['split_out']
                         }

PARAM_DICT['f_theta'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [phi_x_out + phi_z_out + n_ht, 50, n_ht]}
