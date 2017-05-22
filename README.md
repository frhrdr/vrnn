# VRNN Project
Code for an independent study project I did during my Master's, implementing [Variational Recurrent Neural Networks by Chung et al.](https://arxiv.org/abs/1506.02216) in Tensorflow with application to the IAM-OnDB Handwriting data-set and MNIST.

The rough structure is as follows:
- project_report.pdf: explains the model and the data sets I used. (among other things)
- vrnn_model.py: inference and generative model
- vrnn_train.py: training, generation and primed generation functions
- params.py: dictionary specifying all model and training parameters
- utilities.py: helpers for generating networks and some minor other stuff
- vrnn_script.py: execution code
- iamondb_reader.py: preprocessing and plotting functions for the IAM-OnDB handwriting data set
- reference_lstm.py: naive lstm implementation as reference model. only for training and reporting validation error