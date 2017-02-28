PARAM_DICT = dict()

PARAM_DICT['watchlist'] = {'allmc': []}

# data path
PARAM_DICT['series'] = -1
PARAM_DICT['data_path'] = 'data/handwriting/rough_cut_200_pad_500_max_300_norm_xyonly.npy'
PARAM_DICT['log_path'] = 'data/logs/handwriting_10'
PARAM_DICT['log_freq'] = 1000
PARAM_DICT['print_freq'] = 200

# (settle architecture: vanilla, gm_out, gm_latent, multinomial_out, gm_latent_multinomial_out)
# other architectures put on halt
PARAM_DICT['model'] = 'vanilla'
PARAM_DICT['split_latent'] = 1
PARAM_DICT['split_out'] = 1

# specify global settings
PARAM_DICT['batch_size'] = 100
PARAM_DICT['data_dim'] = 2
PARAM_DICT['n_latent'] = 200
PARAM_DICT['seq_length'] = 200
PARAM_DICT['learning_rate'] = 0.0003
PARAM_DICT['max_iter'] = 2000
PARAM_DICT['hid_state_size'] = 2000
PARAM_DICT['masking'] = True
PARAM_DICT['mask_value'] = 500

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
phi_x_out = 200  # 200
phi_z_out = 200  # 200
phi_enc_out = 200  # 200
phi_prior_out = 200  # 200
phi_dec_out = 200  # 200

# specify each net
PARAM_DICT['phi_x'] = {'name': 'phi_x',
                       'nn_type': 'general_mlp',
                       'activation': 'relu',
                       'layers': [n_in, phi_x_out]}

PARAM_DICT['phi_prior'] = {'name': 'phi_prior',
                           'nn_type': 'general_mlp',
                           'activation': 'relu',
                           'layers': [n_ht, phi_prior_out],
                           'out2dist': 'normal',
                           'init_bias': 0.0,
                           'use_batch_norm': False,
                           'splits': PARAM_DICT['split_latent'],
                           'dist_dim': n_z
                           }

PARAM_DICT['phi_enc'] = {'name': 'phi_enc',
                         'nn_type': 'general_mlp',
                         'activation': 'relu',
                         'layers': [phi_x_out + n_ht, phi_enc_out],
                         'out2dist': 'normal',
                         'init_bias': 0.0,
                         'splits': PARAM_DICT['split_latent'],
                         'dist_dim': n_z
                         }

PARAM_DICT['phi_z'] = {'name': 'phi_z',
                       'nn_type': 'general_mlp',
                       'activation': 'relu',
                       'layers': [n_z, phi_z_out]}

PARAM_DICT['phi_dec'] = {'name': 'phi_dec',
                         'nn_type': 'general_mlp',
                         'activation': 'relu',
                         'layers': [phi_z_out + n_ht, phi_dec_out],
                         'out2dist': 'normal',
                         'init_bias': 0.0,
                         'splits': PARAM_DICT['split_out'],
                         'dist_dim': n_out
                         }

PARAM_DICT['f_theta'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [n_ht]}
