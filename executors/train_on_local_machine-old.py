import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from executors.training_utils import *
from dataloader.DataLoader import DataGenerator
from modeling.Transformer import TransformerModel


def train_and_validate(
        # modeling
        model, positional_encodings,
        optimizer, criterion, clip,
        # data
        train_iterator, validation_iterator, 
        # training
        window, stride, init_token, device,n_epochs, 
        # plot validation at the end of an epoch
        n_iters=5, 
        num_plots=5,
        test_samples = None,
        initial_best_valid_loss = float('inf'),
        saving_path:str=None # to be added a saving option
        ): 
    """
    Train a model, validate it, print results and plot signals
    """
    best_valid_loss = initial_best_valid_loss

    for epoch in range(n_epochs):
        
        # take starting time
        start_time = time.time()
        
        # train
        train_loss = train(model, train_iterator, optimizer, criterion, clip, window, stride, init_token,device, n_iters,
                            # for plotting:
                            validation_iterator, positional_encodings, num_plots
                            )
        
        # evaluate
        print('init_token', init_token)
        valid_loss = evaluate(model, validation_iterator, criterion, window,stride,init_token,device, n_iters)

        # take ending time
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        

        # make this part
        print(f"Results of inference on validation, end of epoch {epoch+1}")
        plot_validation_signals(model, validation_iterator, device, init_token, window, positional_encodings, num_plots)
        
        # keep best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if saving_path:
                torch.save(model.state_dict(), f'{saving_path}.pt')

        # print summary
        print('-'*45)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        print('-'*45)
    
        # if test_samples is not None:
        #     print('*'*len('* Results on Test Examples *'))
        #     print('* Results on Test Examples *')
        #     print('*'*len('* Results on Test Examples *'))
        #     plot_test_signals(model, test_samples, num_plots=len(test_samples))


def trainer(seed, batch_size, data_folder_path, lead1, lead2,
         input_dimension, output_dimension, 
         hidden_dimmension, attention_heads, 
         encoder_number_of_layers, decoder_number_of_layers, dropout,clip,
         n_epochs, window, positional_encodings, stride,init_token, 
         # presentation
         n_iters,  # every how many iterations to plot predictions?
         num_plots,
         ):
    
    # Fix randomness
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps")
    print('\n')
    print(f'training using device: {device}')
    print('\n')


    # Load splits dictionary
    # saved_files = os.listdir(data_folder_path) - turn off for now
    with open(data_folder_path+"splits.pkl", 'rb') as handle:
        splits = pickle.load(handle)

    # Create data generators
    train_generator = DataGenerator(lead1, lead2,                        # leads
                                    data_folder_path,                    # path to relevant folder
                                    batch_size,                          # batch size
                                    list_IDs=splits['train'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
                                    shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                                    )
    validation_generator = DataGenerator(lead1, lead2,                        # leads
                                    data_folder_path,                    # path to relevant folder
                                    batch_size,                          # batch size
                                    list_IDs=splits['validation'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
                                    shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                                    )
    test_generator = DataGenerator(lead1, lead2,                        # leads
                                    data_folder_path,                    # path to relevant folder
                                    batch_size,                          # batch size
                                    list_IDs=splits['test'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
                                    shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                                    )
    
    
    # create a model
    model = TransformerModel(input_dimension, output_dimension, 
                        hidden_dimmension, attention_heads, 
                        encoder_number_of_layers, decoder_number_of_layers, dropout,positional_encodings).to(device)

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.MSELoss()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)

    train_and_validate(
        # modeling
        model, positional_encodings,
        optimizer, criterion, clip,
        # data
        train_generator, validation_generator,
        # training
        window, stride, init_token,device, n_epochs, 
        n_iters, 
        # plot validation at the end of an epoch
        num_plots,
        test_samples = None,
        initial_best_valid_loss = float('inf'))
    
    