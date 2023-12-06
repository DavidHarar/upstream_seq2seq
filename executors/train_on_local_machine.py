import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time


from dataloader.DataLoader import DataGenerator, DataGenerator_12leads
from modeling.Transformer import TransformerModel, TSTransformerEncoder, TSTransformerEncoderCNN
from utils.plots import *
from utils import DataCorruptor
from utils.signal_processing import *


import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm.auto import tqdm

from dataloader.DataLoader import process_signals


def trainer(seed, 
            data_folder_path, 
            # architecture
            input_dimension, output_dimension, 
            hidden_dimmension, attention_heads, 
            encoder_number_of_layers, decoder_number_of_layers, 
            positional_encodings, dropout, clip,
            # training
            batch_size, 
            n_epochs, 
            window, 
            train_by_sample,
            saving_path,
            # presentation
            n_iters,                        # every how many iterations to plot predictions?
            model_type,
            plot_saving_path,
            # data corruptor
            scrutiny_probs
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
    
    # init data_corruptor
    data_corruptor = DataCorruptor.DataCorruptor('./data/processed/', scrutiny_probs)

    # Load splits dictionary
    # saved_files = os.listdir(data_folder_path) - turn off for now
    with open(data_folder_path+"splits.pkl", 'rb') as handle:
        splits = pickle.load(handle)

    # Create data generators
    # if lead1 is None:
    train_generator = DataGenerator_12leads(
                            data_folder_path=data_folder_path,   # path to relevant folder
                            batch_size=batch_size,               # batch size
                            list_IDs=splits['train'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
                            shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                            )
    validation_generator = DataGenerator_12leads(
                            data_folder_path=data_folder_path,   # path to relevant folder
                            batch_size=batch_size,               # batch size
                            list_IDs=splits['validation'],       # list of relevat IDs (each ID is given in a <file>_<person> format)
                            shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                            )
    test_generator = DataGenerator_12leads(
                            data_folder_path=data_folder_path,   # path to relevant folder
                            batch_size=batch_size,               # batch size
                            list_IDs=splits['test'],             # list of relevat IDs (each ID is given in a <file>_<person> format)
                            shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
                            )
        

    # create a model
    if model_type == 'autoencoder':
        model = TransformerModel(input_dimension, output_dimension, 
                            hidden_dimmension, attention_heads, 
                            encoder_number_of_layers, decoder_number_of_layers, dropout,positional_encodings).to(device)
        
    if model_type == 'encoder':
        model = TSTransformerEncoder(input_dimension, output_dimension, 
                            hidden_dimmension, attention_heads, 
                            encoder_number_of_layers, positional_encodings,dropout).to(device)
    
    if model_type == 'encoder_cnn':
        model = TSTransformerEncoderCNN(input_dimension, output_dimension, 
                            hidden_dimmension, attention_heads, 
                            encoder_number_of_layers, dropout).to(device)
    

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.MSELoss()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)

    # if we are training a one-lead model
    train_and_validate(
        # modeling
        model, positional_encodings,
        optimizer, criterion, clip,
        # data
        train_generator, validation_generator,
        # training
        window, device, n_epochs, 
        n_iters, saving_path, train_by_sample,
        # numner of electrods: 1 lead or 12 leads model?
        output_dimension, data_corruptor,
        # plot validation at the end of an epoch
        plot_saving_path,
        initial_best_valid_loss = float('inf'),
        )
        


def train_and_validate(
        # modeling
        model, positional_encodings,
        optimizer, criterion, clip,
        # data
        train_iterator, validation_iterator,
        # training
        window, device, n_epochs, 
        n_iters, saving_path, train_by_sample,
        # numner of electrods: 1 lead or 12 leads model?
        output_dimension, data_corruptor,
        # plot validation at the end of an epoch
        plot_saving_path,
        initial_best_valid_loss = float('inf'),
        ): 
    """
    Train a model, validate it, print results and plot signals
    """
    best_valid_loss = initial_best_valid_loss
    losses = {'train':[],
              'validation':[]}

    for epoch in range(n_epochs):
        
        # take starting time
        start_time = time.time()
        
        # train
        train_loss = train(model, 
                           train_iterator, 
                           optimizer, 
                           criterion, 
                           clip, 
                           window, 
                           device, 
                           n_iters,
                           # for plotting:
                           validation_iterator,
                           # training paradigm is defined by output_dimension
                           data_corruptor,
                           # Training scheme
                           train_by_sample
                           )
        


        # evaluate
        valid_loss = evaluate(model, 
                              validation_iterator, 
                              criterion, 
                              window,
                              device,
                              data_corruptor
                              )

        # store losses
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)

        with open(f'{saving_path}_loss.pkl', 'wb') as f:
            pickle.dump(losses, f)
        

        # take ending time
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        

        # make this part
        if plot_saving_path:
            plot_validation_signals_12leads(model, 
                                validation_iterator, 
                                data_corruptor,
                                device, 
                                window,
                                f'{plot_saving_path}_epoch-{epoch}')
            
        # keep best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if saving_path:
                torch.save(model.state_dict(), f'{saving_path}_{time.time()}.pt')

        # print summary
        print('-'*45)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        print('-'*45)
    


def train(
        # model params
        model, 
        iterator, 
        optimizer, 
        criterion, 
        clip, 
        window, 
        device, 
        n_iters,
        # plotting params
        validation_iterator, 
        # training paradigm is defined by output_dimension
        data_corruptor,
        # train by sample or progress over the entire signal
        train_by_sample:int
        ):
    
    # set model on training state and init epoch loss    
    model.train()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. 
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (training)', leave=True)

    for i in t:
        # get data
        src, trg = next(it)
        # validate data
        assert src.shape==trg.shape, "Source signal and target signal have to have the same length."

        if np.sum(src!=src):
            continue
        
        for _ in range(train_by_sample):
            
            # process signals
            src_ = resample(src)
            trg_ = resample(trg)


            # sample
            src_, trg_ = sample(src_, trg_, window=window)

            src_ = normalize(src_)
            trg_ = normalize(trg_)

            # data corruption
            src_ = data_corruptor.corrupt_a_batch(src_)

            # fix shapes (convert into -> [length, batch_size, channels])
            src_ = np.float32(np.transpose(src_, axes=(2,0,1)))
            trg_ = np.float32(np.transpose(trg_, axes=(2,0,1)))
            
            # don't run if there are NaNs
            if np.isnan(src_).sum()>0:
                print('skipping because of NaNs')
                continue

            src_ = torch.from_numpy(src_)
            trg_ = torch.from_numpy(trg_)

            src_ = src_.to(device)
            trg_ = trg_.to(device)
            
            # step
            optimizer.zero_grad()
            output = model(src_, trg_)

            loss = criterion(output, trg_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

            j = np.round(epoch_loss/(i+1),5)
            t.set_description(f"Within epoch loss (training) {j}")
            t.refresh() # to show immediately the update

            # update values
            denominator+=1


        if n_iters is not None:
            if (i%n_iters)==0 and i>0:
                print(f"Intermediate results of inference on training, iteration {i}")
                plot_validation_signals_12leads(model, 
                                    iterator, 
                                    data_corruptor,
                                    device, 
                                    window)
                print(f"Intermediate results of inference on validation, iteration {i}")
                plot_validation_signals_12leads(model, 
                                    validation_iterator, 
                                    data_corruptor,
                                    device, 
                                    window)

    return epoch_loss / denominator

def evaluate(model, 
             iterator, 
             criterion,
             window,
             device,
             data_corruptor
    ):
    # set model on training state and init epoch loss    
    model.eval()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. n_iters can be set to bypass it
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (validation)', leave=True)

    with torch.no_grad():
        for i in t:

            # get data
            src, trg = next(it)

            # process signals
            src_ = resample(src)
            trg_ = resample(trg)

            # validate data
            assert src.shape==trg.shape, "Source signal and target signal have to have the same length."
            
            src_, trg_ = sample(src_, trg_, window=window, fix_start=25)
            src_ = normalize(src_)
            trg_ = normalize(trg_)

            src_ = data_corruptor.corrupt_a_batch(src_)

            # fix shapes (convert into -> [length, batch_size, channels])
            src_ = np.float32(np.transpose(src_, axes=(2,0,1)))
            trg_ = np.float32(np.transpose(trg_, axes=(2,0,1)))
            
            # don't run if there are NaNs
            if np.isnan(src_).sum()>0:
                print('skipping because of NaNs')
                continue

            src_ = torch.from_numpy(src_)
            trg_ = torch.from_numpy(trg_)

            src_ = src_.to(device)
            trg_ = trg_.to(device)

            output = model(src_, trg_)
            loss = criterion(output, trg_)
            epoch_loss += loss.item()

            j = np.round(epoch_loss/(i+1),5)
            t.set_description(f"Within epoch loss (validation) {j}")
            t.refresh() # to show immediately the update

            # update values
            denominator+=1
        
    return epoch_loss / denominator


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)