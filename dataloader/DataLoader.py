import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import List

def process_signals(signal,init_token,mode='train', model_type = 'autoencoder'):
    if model_type=='autoencoder':
        if mode=='train':
            if init_token is not None:
                signal = np.insert(signal, obj=0, values = init_token, axis=0)
                init_for_eval_stage = init_token
            else:
                init_for_eval_stage=signal[0,:]
                signal = np.insert(signal, obj=0, values = init_for_eval_stage, axis=0)

        if mode=='eval':
            # insert SOS at the beggining
            signal = np.insert(signal, obj = 0, values = init_token, axis=0)
            init_for_eval_stage=None
    else:
        init_for_eval_stage=None

    # expand dim to be compatible with torch models
    signal_ = np.expand_dims(signal,-1)
    
    # convert to torch
    signal_ = torch.from_numpy(signal_).float()

    return signal_, init_for_eval_stage



class DataGenerator():
    def __init__(self,
                 lead1:str, lead2:str, # leads
                 data_folder_path:str, # path to relevant folder
                 batch_size:int,       # batch size
                 list_IDs:List[str],   # list of relevat IDs (each ID is given in a <file>_<person> format)
                 shuffle:bool = True   # Whether to shuffle the list of IDs at the end of each epoch.
                 ):
        
        self.data_path = data_folder_path
        self.lead1 = lead1
        self.lead2 = lead2
        self.scaler = MinMaxScaler()
        self.list_IDs = list_IDs
        self.indices = np.arange(len(self.list_IDs))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_sample(self, lead, person, data_folder_path):
        """
        load the relevant <lead> reading .npy file from <data_folder_path>. 
        Inputs:
        - lead:str. The lead we want to load.
        - person:str. The index of the person.
        - data_folder_path:str. The path to the folder when the desired file can be loaded from
        """

        file_name = f'{person.split("_")[0]}_{lead}_{person.split("_")[1]}.npy'
        signal = np.load(data_folder_path+file_name,allow_pickle=True)
        
        # fillna in signal by mean, and take the int of it
        signal[signal!=signal] = int(np.nanmean(signal))

        return signal

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        import random
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    
    def __data_generation(self, list_IDs_temp):
        """
        Get a list of persons IDs and return a batch of X1 and X2 signals.
        """
        lead1 = self.lead1
        lead2 = self.lead2
        data_path = self.data_path

        # Initialization
        X1 = np.array([self._get_sample(lead1, person_id, data_path) for person_id in list_IDs_temp])
        X2 = np.array([self._get_sample(lead2, person_id, data_path) for person_id in list_IDs_temp])
        # X1,X2 = [],[]
        # for person_id in list_IDs_temp:
        #     print(lead1, person_id, data_path), print(lead1, person_id, data_path)
        #     X1.append(self._get_sample(lead1, person_id, data_path))
        #     X2.append(self._get_sample(lead2, person_id, data_path))
        # X1 = np.array(X1)
        # X2 = np.array(X2)
        
        return X1, X2
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def __getitem__(self, index):
        'Generate one batch of data using __data_generation'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X1, X2 = self.__data_generation(list_IDs_temp)
        
        # cast to int
        X1=X1.astype(int)
        X2=X2.astype(int)

        # normalize
        X1 = self.scaler.fit_transform(X1.T).T
        X2 = self.scaler.fit_transform(X2.T).T

        return X1, X2


# --------------------------------
# Usage example
# --------------------------------
# DESTINATION_FOLDER = '../data/processed/'
# LEAD1='LI'
# LEAD2='aVF'
# BATCH_SIZE = 34
# with open(DESTINATION_FOLDER+"splits.pkl", 'rb') as handle:
#     splits = pickle.load(handle)

# train_generator = DataGenerator(lead1=LEAD1, lead2=LEAD2,             # leads
#                                 data_folder_path=DESTINATION_FOLDER, # path to relevant folder
#                                 batch_size=BATCH_SIZE,                        # batch size
#                                 list_IDs=splits['train'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
#                                 shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
#                                 )

# X1,X2 = train_generator.__getitem__(0)
# print(X1.shape, X2.shape)
# X1, X2





class DataGenerator_12leads():
    def __init__(self,
                 data_folder_path:str,  # path to relevant folder
                 batch_size:int,        # batch size
                 list_IDs:List[str],    # list of relevat IDs (each ID is given in a <file>_<person> format)
                 shuffle:bool = True,   # Whether to shuffle the list of IDs at the end of each epoch.
                 normalize:bool = False
                 ):
        
        self.data_path = data_folder_path
        self.leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6']
        self.scaler = MinMaxScaler()
        self.list_IDs = list_IDs
        self.indices = np.arange(len(self.list_IDs))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
    


    def _get_sample(self, lead, person, data_folder_path):
        """
        load the relevant <lead> reading .npy file from <data_folder_path>. 
        Inputs:
        - lead:str. The lead we want to load.
        - person:str. The index of the person.
        - data_folder_path:str. The path to the folder when the desired file can be loaded from
        """

        file_name = f'{person.split("_")[0]}_{lead}_{person.split("_")[1]}.npy'
        signal = np.load(data_folder_path+file_name,allow_pickle=True)
        
        # fillna in signal by mean, and take the int of it
        signal[signal!=signal] = int(np.nanmean(signal))

        return signal
    
    def normalize_12_leads(self, leads_signals,scaler):
        return [scaler.fit_transform(leads_signals[j].reshape((-1,1))) if j==0 else scaler.transform(leads_signals[j].reshape((-1,1))) for j in range(12)]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        import random
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    
    def __data_generation(self, list_IDs_temp):
        """
        Get a list of persons IDs and return a batch of X1 and X2 signals.
        """
        data_path = self.data_path

        # Initialization
        X1 = np.array([[self._get_sample(lead, person_id, data_path) for lead in self.leads] for person_id in list_IDs_temp])
        
        return X1
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def __getitem__(self, index):
        'Generate one batch of data using __data_generation'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X1 = self.__data_generation(list_IDs_temp)
        
        # cast to int
        X1=X1.astype(int)

        # normalize
        if self.normalize:
            X1 = X2 = np.array([self.normalize_12_leads(X1[j],self.scaler) for j in range(X1.shape[0])]).squeeze(-1)
        else:
            X2=X1
        return X1, X2
    




# --------------------------------
# Usage example
# --------------------------------
# import pickle
# DESTINATION_FOLDER = './data/processed/'
# BATCH_SIZE = 4
# with open(DESTINATION_FOLDER+"splits.pkl", 'rb') as handle:
#     splits = pickle.load(handle)

# train_generator = DataGenerator_12leads(
#                                 data_folder_path=DESTINATION_FOLDER, # path to relevant folder
#                                 batch_size=BATCH_SIZE,                        # batch size
#                                 list_IDs=splits['train'],            # list of relevat IDs (each ID is given in a <file>_<person> format)
#                                 shuffle = True                       # Whether to shuffle the list of IDs at the end of each epoch.
#                                 )

# X1,X2 = train_generator.__getitem__(0)
# print(X1.shape, X2.shape)
# X1, X2