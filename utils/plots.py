import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from dataloader.DataLoader import DataGenerator, process_signals
from modeling.Transformer import TransformerModel, TSTransformerEncoder
from utils.plots import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm.auto import tqdm
from utils.signal_processing import *

def plot_validation_signals(model, validation_iterator, device, init_token,window, positional_encodings,model_type, num_plots=5, path=None)->None:
    """
    Plot actual vs. predicted. Results are a plot of `2 x <num_plots>`, first column
    contains plots of the source signals. Second column contains plots of the target
    signals, and their prediction.
    """


    # Choose a random batch to take. Search over the first 10 for time efficiency
    it = iter(validation_iterator)
    T=1
    t = trange(T)
    rand = 1 #np.random.randint(10)
    with torch.no_grad():
        for i in t:
            # get data
            src, trg = next(it)    
        
            # stopping creterea
            if i == rand:
                break

        # transpose
        src=src.T
        trg=trg.T
        
        # validate data
        assert len(src)==len(trg), "Source signal and target signal have to have the same length."

        # start training
        start = np.random.randint(len(src)-window-1)

        src_ = src[start:(start+window)]
        trg_ = trg[start:(start+window)]
        
        # fix shapes
        src_, init_for_eval_stage = process_signals(src_,init_token, mode='train',model_type=model_type)
        trg_, init_for_eval_stage = process_signals(trg_,init_for_eval_stage, mode='eval',model_type=model_type)
               
        src_ = src_.to(device)
        trg_ = trg_.to(device)
            
        batch_size = src.shape[1]
        
        # plot
        N_plots = min(batch_size,num_plots)
        fig, axs = plt.subplots(N_plots, 2, figsize=(12,18))
        if model_type=='autoencoder':
            for i in range(N_plots):
                # print('before', type(src_))
                source_signal = src_[:,i,:].unsqueeze(-1)
                x = src_[:,i,:].unsqueeze(-1).cpu().numpy()
                y = trg_[:,i,:].unsqueeze(-1).cpu().numpy()
                # print('after', type(x))
                # pred
                y_pred = predict_sequence_for_a_single_signal(model,source_signal, device, positional_encodings, init_token, model_type)

                
                # reshape
                x = x.reshape(-1)
                y = y.reshape(-1)
                y_pred = y_pred.reshape(-1)

                
                axs[i, 0].plot(x)       # plot source signal on the left column at position [i]
                axs[i, 1].plot(y)       # plot target signal on the right column at position [i]
                axs[i, 1].plot(y_pred)  # plot predicted signal on the right column at position [i]
        if model_type=='encoder':
            y_pred = model(src_,None)
            for i in range(N_plots):
                # print('before', type(src_))
                source_signal = src_[:,i,:].unsqueeze(-1)
                x = src_[:,i,:].unsqueeze(-1).cpu().numpy()
                y = trg_[:,i,:].unsqueeze(-1).cpu().numpy()
                y_pred_i = y_pred[:,i,:].unsqueeze(-1).cpu().numpy()
                                
                # reshape
                x = x.reshape(-1)
                y = y.reshape(-1)
                y_pred_i = y_pred_i.reshape(-1)

                
                axs[i, 0].plot(x)       # plot source signal on the left column at position [i]
                axs[i, 1].plot(y)       # plot target signal on the right column at position [i]
                axs[i, 1].plot(y_pred_i)  # plot predicted signal on the right column at position [i]

        if path is not None:
            if not path.endswith('.png'):
                path += '.png'
            fig.savefig(path)
            plt.close(fig)
        else:
            plt.show()



def plot_validation_signals_12leads(model, 
                                    validation_iterator, 
                                    data_corruptor,
                                    device, 
                                    window,
                                    plot_saving_path=None)->None:
    """
    Plot actual vs. predicted. Results are a plot of `2 x <num_plots>`, first column
    contains plots of the source signals. Second column contains plots of the target
    signals, and their prediction.
    """


    # Choose a random batch to take. Search over the first 10 for time efficiency
    it = iter(validation_iterator)
    T=1
    t = trange(T)
    with torch.no_grad():
        for i in t:
            # get data from the first batch
            src, trg = next(it)    
            if i == 1:
                break
        
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
        
        # move to gpu so we can pass through model
        src_ = torch.from_numpy(src_).to(device)
        trg_ = trg_ #.to(device)

        # choose which signal from the first batch to look at
        batch_size = src.shape[1]
        sig_to_plot = np.random.randint(0,batch_size,1)

        # predict
        y_pred = model(src_,None).cpu().numpy()

        # prep signals for presentation
        original = np.squeeze(trg_[:,sig_to_plot,:]).T
        corrupted = np.squeeze(src_[:,sig_to_plot,:].cpu().numpy()).T
        reconstructed = np.squeeze(y_pred[:,sig_to_plot,:]).T
        
        plot_12_signals_of_three_sources(original,corrupted, reconstructed, plot_saving_path)

def plot_12_signals_of_three_sources(reading, corrupted, predicted, plot_saving_path):
    """
    plot a 12 electrodes reading of before/after. For example, before and after corruption,
    and actutal vs. predicted.
    """
    fig, axs = plt.subplots(6, 2, figsize = (20,20))

    leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6']
    s=0
    for ax in axs.flat:
        ax.plot(reading[s])
        ax.plot(corrupted[s])
        ax.plot(predicted[s])
        ax.set_title(leads[s],fontsize=9)
        s+=1
    fig.suptitle('Blue-Original; Orange-Corrupted; Greed-Predicted')
    plt.tight_layout()
    if plot_saving_path:
        plt.savefig(f'{plot_saving_path}.png')
    plt.show()


def predict_sequence_for_a_single_signal(model,src,device, positional_encodings, init_token, model_type):
    """
    Given a trained transformer model and a source signal, return a new signal of the target lead.


    LATER: ADD an option to hundle multiple signals at once, start from here
    trg_tensor = torch.tensor(np.repeat(out_indexes, source_signal.shape[1])).unsqueeze(0).unsqueeze(-1).to(device)
    output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))

    """
    def __get_output(output):
        values = output.cpu().numpy().squeeze()
        if len(values.shape)==0:
            return float(values)
        else:
            return values[-1]
    
    assert src.shape[1] == 1, "A signle signal must be entered."
    
    # get length
    seq_length = src.shape[0]

    if len(src)==2:
        src=src.unsqueeze(-1)

    # extract sos token
    if not init_token:
        if torch.is_tensor(src):
            init_token = src.cpu().numpy().reshape(-1)[0]
        else:
            init_token = src.numpy().reshape(-1)[0]
        seq_length = src.shape[0]
        # seq_length = seq_length-1 # account for initial token that is already in src
    
    # store source signal in memory
    if model_type=='autoencoder':
        if positional_encodings:
            memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))
        else:
            memory = model.transformer.encoder(model.encoder(src))
        
        # init a response sequence
        out_indexes = [init_token, ]

        # for i in range(max_len): # we have constant length, same as seq_length (after removing sos)
        for i in tqdm(range(seq_length-1)):
            if len(out_indexes)==1:
                trg_tensor = torch.tensor(out_indexes).unsqueeze(-1).to(device)
            else:
                trg_tensor = torch.tensor(out_indexes).unsqueeze(-1).unsqueeze(-1).to(device)

            if positional_encodings:
                # print('with pos', model.pos_decoder(model.decoder(trg_tensor)).shape, memory.shape)
                decoded_values=model.pos_decoder(model.decoder(trg_tensor))
            else:
                decoded_values=model.decoder(trg_tensor)
                # print('without pos', model.decoder(trg_tensor).shape, memory.shape)
                if len(decoded_values.shape) == 2:
                    decoded_values = decoded_values.unsqueeze(0)
            
            output = model.fc_out(model.transformer.decoder(decoded_values, memory))

            out_indexes.append(__get_output(output))
            
    if model_type=='encoder':
        # init a response sequence
        out_indexes = [init_token, ]
        trg_tensor = torch.tensor(out_indexes).unsqueeze(-1).unsqueeze(-1).to(device)
        res = model(src,trg_tensor)
        print('zain baaiin', res.shape)
        out_indexes = __get_output(res)
    
    return np.array(out_indexes)
