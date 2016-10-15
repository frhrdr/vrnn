from vrnn_train import run_training, run_generation
from params import PARAM_DICT

# run_training(PARAM_DICT)
x = run_generation('data/logs/test1/params.pkl')

print x
