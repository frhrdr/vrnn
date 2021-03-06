PARAM_DICT = dict()

# data path
PARAM_DICT['series'] = -1
# PARAM_DICT['train_data_path'] = 'data/handwriting/rough_cut_500_pad_500_max_300_norm.npy'
PARAM_DICT['train_data_path'] = 'load_mnist'
PARAM_DICT['log_path'] = 'data/logs/mnist_23/'
PARAM_DICT['log_freq'] = 500
PARAM_DICT['print_freq'] = 500
PARAM_DICT['valid_freq'] = 500
PARAM_DICT['load_path'] = 'data/logs/mnist_19/ckpt-20000'
PARAM_DICT['num_ckpts'] = 5
PARAM_DICT['validation_set_size'] = 10000
PARAM_DICT['valid_data_path'] = 'data/handwriting/rough_cut_500_pad_500_max_300_norm.npy'

# specify global settings
PARAM_DICT['model'] = 'gauss_out'  # options: gauss_out, gm_out, gauss_out_bin, gm_out_bin
PARAM_DICT['modes_out'] = 1
PARAM_DICT['batch_size'] = 100
PARAM_DICT['x_dim'] = 28
PARAM_DICT['z_dim'] = 2
PARAM_DICT['seq_length'] = 28
PARAM_DICT['learning_rate'] = 0.000003
PARAM_DICT['max_iter'] = 20000
PARAM_DICT['hid_state_size'] = 1500
PARAM_DICT['masking'] = False
PARAM_DICT['mask_value'] = 500
PARAM_DICT['kl_weight'] = 1.0

# infer some necessary network sizes
PARAM_DICT['in_dim'] = PARAM_DICT['x_dim']

if 'gauss' in PARAM_DICT['model']:
    out_dist = 'gauss'
    PARAM_DICT['modes_out'] = 1
else:
    out_dist = 'gm'

if 'bin' in PARAM_DICT['model']:
    out_dist += '_plus_bin'
    PARAM_DICT['in_dim'] += 1

# assign shared variables
phi_x_out = 300
phi_z_out = 300
phi_enc_out = 300
phi_prior_out = 300
phi_dec_out = 300

# specify each net
PARAM_DICT['phi_x'] = {'name': 'phi_x',
                       'nn_type': 'general_mlp',
                       'activation': 'relu',
                       'layers': [PARAM_DICT['in_dim'], 300, phi_x_out]}

PARAM_DICT['phi_prior'] = {'name': 'phi_prior',
                           'nn_type': 'general_mlp',
                           'activation': 'relu',
                           'layers': [PARAM_DICT['hid_state_size'], 300, phi_prior_out],
                           'out2dist': 'gauss',
                           'init_sig_var': 0.01,
                           'init_sig_bias': 0.0,
                           'dist_dim':  PARAM_DICT['z_dim']
                           }

PARAM_DICT['phi_enc'] = {'name': 'phi_enc',
                         'nn_type': 'general_mlp',
                         'activation': 'relu',
                         'layers': [phi_x_out + PARAM_DICT['hid_state_size'], 300, phi_enc_out],
                         'out2dist': 'gauss',
                         'init_sig_var': 0.01,
                         'init_sig_bias': 0.0,
                         'dist_dim':  PARAM_DICT['z_dim']
                         }

PARAM_DICT['phi_z'] = {'name': 'phi_z',
                       'nn_type': 'general_mlp',
                       'activation': 'relu',
                       'layers': [PARAM_DICT['z_dim'], 300, phi_z_out]}

PARAM_DICT['phi_dec'] = {'name': 'phi_dec',
                         'nn_type': 'general_mlp',
                         'activation': 'relu',
                         'layers': [phi_z_out + PARAM_DICT['hid_state_size'], 300, phi_dec_out],
                         'out2dist': out_dist,
                         'init_sig_var': 0.01,
                         'init_sig_bias': 0.0,
                         'modes': PARAM_DICT['modes_out'],
                         'dist_dim':  PARAM_DICT['x_dim']
                         }

PARAM_DICT['f_theta'] = {'name': 'f_theta',
                         'nn_type': 'general_lstm',
                         'layers': [PARAM_DICT['hid_state_size'], PARAM_DICT['hid_state_size']]}


