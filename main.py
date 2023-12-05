
# Train a seq2seq model from one ECG lead to another.
import random
import json
import pickle

# import dataloader.DataLoader
from executors.train_on_local_machine_mps import trainer
# import modeling.Transformer

# Load config file
config = {
    # general
    'seed':123,
    'data_folder_path': '../../dissertation/upstream_data/',
    'lead1':"LI", 'lead2':"LII",
    # training
    'batch_size':32,
    'n_epochs':10,
    # architecture - to be changed later and pushed out towards tuning
    'init_token': 0,
    'input_dimension':1, 'output_dimension':1,  # these should remain constant
    'hidden_dimmension': 64,                               # d_model (int) – the number of expected features in the input (required)???
    'attention_heads':None,                         # number of attention heads, if None then d_model//64
    'encoder_number_of_layers':3,
    'decoder_number_of_layers':3,
    'dropout':0.1,
    'clip':1,
    'window':500, 
    'stride':0.1,
    'n_iters': 15, # number of signals to train on. All if None
    'num_plots':5,

}


# run
trainer(**config)


