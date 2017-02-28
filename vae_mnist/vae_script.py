from params import *
from vae_train import *
from vrnn_test import utilities as util

decoder_fun = lambda t_in, in_dims, n_out: util.simple_mlp(t_in, in_dims, DEFAULT_N_HIDDEN, n_out, 'decoder', [])

# run_training(encoder_fun=None, decoder_fun=decoder_fun)

run_generation('data/logs/test1/checkpoint-8999', decoder_fun, N_LATENT, IMG_DIMS, 64)
