readme, refactoring, basic tutorial in progress...

# Project structure

### VAE
To be reworked or deleted. Probably not working at this point.

### VRNN
- vrnn_model.py: inference and generative model
- vrnn_train.py: training, generation and primed generation functions
- params.py: dictionary specifying all model an training parameters
- vrnn_script.py: execution code
- iamondb_reader.py: preprocessing and plotting functions for the IAM-OnDB handwriting data set
- reference_lstm.py: naive lstm implementation as reference model. only for training and reporting validation error