import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm.auto import tqdm

from dataloader.DataLoader import process_signals


def train(model, iterator, optimizer, criterion, clip, window, stride, init_token,device, n_iters,model_type,
          # for plotting:
          validation_iterator, positional_encodings, num_plots,
          ):
    
    # set model on training state and init epoch loss    
    model.train()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. n_iters can be set to bypass it
    it = iter(iterator)
    # if n_iters:
    #     T = min(len(iterator),n_iters)
    # else:
    #     T = len(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (training)', leave=True)

    for i in t:
        # # stopping creterea
        # if i == n_iters:
        #     break

        # get data
        src, trg = next(it)
        
        # transpose
        src=src.T
        trg=trg.T
        
        # validate data
        assert len(src)==len(trg), "Source signal and target signal have to have the same length."

        # if stride is given as a portion. These two lines will run only once
        if stride<=1:
            stride = int(stride*len(src))
        
        # start training
        start = 0 
        while start+window<=len(src):
            src_ = src[start:(start+window)]
            trg_ = trg[start:(start+window)]
            
            # fix shapes
            src_, init_for_eval_stage = process_signals(src_,init_token, mode='train', model_type=model_type)
            trg_, init_for_eval_stage = process_signals(trg_,init_for_eval_stage, mode='eval', model_type=model_type)
            
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
            start += stride
            denominator+=1

        if (i%n_iters)==0:
            print(f"Intermediate results of inference on validation, iteration {i}")
            plot_validation_signals(model, validation_iterator, device, init_token, window, positional_encodings, model_type, num_plots)

    return epoch_loss / denominator

def evaluate(model, iterator, criterion,window,stride, init_token, device,model_type,n_iters = None):
    # set model on training state and init epoch loss    
    model.eval()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. n_iters can be set to bypass it
    it = iter(iterator)
    if n_iters:
        T = min(len(iterator),n_iters)
    else:
        T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (validation)', leave=True)

    with torch.no_grad():
        for i in t:
            # stopping creterea
            if i == n_iters:
                break

            # get data
            src, trg = next(it)
            
            # transpose
            src=src.T
            trg=trg.T
            
            # validate data
            assert len(src)==len(trg), "Source signal and target signal have to have the same length."

            # if stride is given as a portion. It will run only once
            if stride<=1:
                stride = int(stride*len(src))
            
            # start training
            start = 0 

            while start+window<=len(src):
                src_ = src[start:(start+window)]
                trg_ = trg[start:(start+window)]
                
                # fix shapes
                src_, init_for_eval_stage = process_signals(src_,init_token, mode='train', model_type=model_type)
                trg_, init_for_eval_stage = process_signals(trg_,init_for_eval_stage, mode='eval', model_type=model_type)
                       
                src_ = src_.to(device)
                trg_ = trg_.to(device)

                output = model(src_, trg_)
                loss = criterion(output, trg_)
                epoch_loss += loss.item()

                j = np.round(epoch_loss/(i+1),5)
                t.set_description(f"Within epoch loss (validation) {j}")
                t.refresh() # to show immediately the update

                # update values
                start += stride
                denominator+=1
        
    return epoch_loss / denominator


def predict_sequence_for_a_single_signal_encoder(model,src):
    def __get_output(output):
        values = output.detach().numpy().squeeze()
        if len(output.detach().numpy().squeeze().shape)==0:
            return float(values)
        else:
            return values[-1]
    
    assert src.shape[1] == 1, "A signle signal must be entered."
    output = model(src)
    output = __get_output.append(output)




def predict_sequence_for_a_single_signal(model,src,device, positional_encodings, init_token):
    """
    Given a trained transformer model and a source signal, return a new signal of the target lead.


    LATER: ADD an option to hundle multiple signals at once, start from here
    trg_tensor = torch.tensor(np.repeat(out_indexes, source_signal.shape[1])).unsqueeze(0).unsqueeze(-1).to(device)
    output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))

    """
    def __get_output(output):
        values = output.detach().numpy().squeeze()
        if len(output.detach().numpy().squeeze().shape)==0:
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
        init_token = src.numpy().reshape(-1)[0]
        seq_length = src.shape[0]
        # seq_length = seq_length-1 # account for initial token that is already in src
    
    # store source signal in memory
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
        
    return np.array(out_indexes)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_validation_signals(model, 
                            validation_iterator, 
                            device, 
                            init_token,window,
                            positional_encodings, 
                            model_type,
                            num_plots=5, path=None)->None:
    """
    Plot actual vs. predicted. Results are a plot of `2 x <num_plots>`, first column
    contains plots of the source signals. Second column contains plots of the target
    signals, and their prediction.
    """


    # Choose a random batch to take. Search over the first 10 for time efficiency
    it = iter(validation_iterator)
    T=11
    t = trange(T)
    rand = np.random.randint(10)
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
        for i in range(N_plots):
            source_signal = src_[:,i,:].unsqueeze(-1)
            x = src_[:,i,:].unsqueeze(-1).detach().numpy()
            y = trg_[:,i,:].unsqueeze(-1).detach().numpy()
            
            # pred
            if model_type == 'autoencoder':
                y_pred = predict_sequence_for_a_single_signal(model,source_signal, device, positional_encodings, init_token)
            if model_type=='encoder':
                pass

            
            # reshape
            x = x.reshape(-1)
            y = y.reshape(-1)
            y_pred = y_pred.reshape(-1)

            
            axs[i, 0].plot(x)       # plot source signal on the left column at position [i]
            axs[i, 1].plot(y)       # plot target signal on the right column at position [i]
            axs[i, 1].plot(y_pred)  # plot predicted signal on the right column at position [i]
            
        if path is not None:
            if not path.endswith('.png'):
                path += '.png'
            fig.savefig(path)
            plt.close(fig)
        else:
            plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


