PARAM_DICT = dict()

# data path
PARAM_DICT['series'] = -1
PARAM_DICT['data_path'] = 'data/handwriting/rough_cut_200_pad_500_max_300_norm_xyonly.npy'
PARAM_DICT['log_path'] = 'data/logs/handwriting_35'
PARAM_DICT['log_freq'] = 500
PARAM_DICT['print_freq'] = 200

# other architectures put on halt
PARAM_DICT['model'] = 'gauss_out'  # options: gauss_out, gm_out
PARAM_DICT['modes_out'] = 1

# specify global settings
PARAM_DICT['batch_size'] = 100
PARAM_DICT['x_dim'] = 2
PARAM_DICT['z_dim'] = 200
PARAM_DICT['seq_length'] = 200
PARAM_DICT['learning_rate'] = 0.001
PARAM_DICT['max_iter'] = 2000
PARAM_DICT['hid_state_size'] = 2000
PARAM_DICT['masking'] = True
PARAM_DICT['mask_value'] = 500

# infer some necessary network sizes
n_in = PARAM_DICT['x_dim']           # x
n_out = PARAM_DICT['x_dim']          # x
n_z = PARAM_DICT['z_dim']            # z
n_ht = PARAM_DICT['hid_state_size']     # h_t

if PARAM_DICT['model'] == 'gauss_out':
    out_dist = 'normal'
    PARAM_DICT['modes_out'] = 1
else:
    out_dist = 'gm'

# assign shared variables
phi_x_out = 200  # 200
phi_z_out = 200  # 200
phi_enc_out = 200  # 200
phi_prior_out = 200  # 200
phi_dec_out = 200  # 200

# specify each net
PARAM_DICT['phi_x'] = {'name': 'phi_x',
                       'nn_type': 'general_mlp',
                       'activation': 'elu',
                       'layers': [n_in, phi_x_out]}

PARAM_DICT['phi_prior'] = {'name': 'phi_prior',
                           'nn_type': 'general_mlp',
                           'activation': 'elu',
                           'layers': [n_ht, phi_prior_out],
                           'out2dist': 'normal',
                           'init_sig_var': 0.01,
                           'init_sig_bias': 0.0,
                           'dist_dim': n_z
                           }

PARAM_DICT['phi_enc'] = {'name': 'phi_enc',
                         'nn_type': 'general_mlp',
                         'activation': 'elu',
                         'layers': [phi_x_out + n_ht, phi_enc_out],
                         'out2dist': 'normal',
                         'init_sig_var': 0.01,
                         'init_sig_bias': 0.0,
                         'dist_dim': n_z
                         }

PARAM_DICT['phi_z'] = {'name': 'phi_z',
                       'nn_type': 'general_mlp',
                       'activation': 'elu',
                       'layers': [n_z, phi_z_out]}

PARAM_DICT['phi_dec'] = {'name': 'phi_dec',
                         'nn_type': 'general_mlp',
                         'activation': 'elu',
                         'layers': [phi_z_out + n_ht, phi_dec_out],
                         'out2dist': out_dist,
                         'init_sig_var': 0.01,
                         'init_sig_bias': 0.0,
                         'modes': PARAM_DICT['modes_out'],
                         'dist_dim': n_out
                         }

PARAM_DICT['f_theta'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [n_ht]}
