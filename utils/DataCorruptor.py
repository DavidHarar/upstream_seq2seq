import random
import numpy as np
import os


class DataCorruptor():
    def __init__(self,
                 processed_data_folder
                 ):
        
        # load noisy signals
        files = [x for x in os.listdir(processed_data_folder) if x.startswith('noise')]

        MA_noise = [x for x in files if '_MA_' in x]
        BW_noise = [x for x in files if '_BW_' in x]
        EM_noise = [x for x in files if '_EM_' in x]
        
        # store results
        noise_signals = {}
        for noise_type in [MA_noise, BW_noise, EM_noise]:
            noise_str = noise_type[0].split('_')[1]
            noise_long_file = []
            for noise_file in noise_type:
                noise_long_file += np.load(processed_data_folder+noise_file,allow_pickle=True).tolist()
            noise_signals[noise_str]=noise_long_file
        
        self.noise_signals=noise_signals
        self.noise_signal_len = len(noise_signals[noise_str])

        # define probabilities for scrutiny types.
        self.scrutiny_probs = \
            [None]*31+\
            ['Turn-off']*20+\
            ['MA']*7+\
            ['BW']*7+\
            ['EM']*7+\
            ['MA+BW']*7+\
            ['MA+EM']*7+\
            ['BW+EM']*7+\
            ['MA+BW+EM']*7

    def corrupt_with_prob(self,signal):
        noise_type = random.choice(self.scrutiny_probs)
        if noise_type=='Turn-off':
            signal = np.zeros_like(signal)
        elif noise_type is not None:
            SNR = random.uniform(1,10)
            signal = self.forward(signal,noise_type, SNR)
        return signal

    def corrupt_a_batch(self, batch):
        """
        Assuming batch is of shape [batch, leads, len]
        """
        
        def corrupt_an_observation(observation):
            """
            Assuminng observation is given by [leads,len]
            """
            return [self.corrupt_with_prob(observation[j]) for j in range(observation.shape[0])]

        return np.array([corrupt_an_observation(batch[j]) for j in range(batch.shape[0])])


    def create_linear_combination(self, signal, noise, SNR):
        """Creates a linear combination of a signal and noise, given a signal to noise ratio.

        Args:
            signal: A numpy array representing the signal.
            noise: A numpy array representing the noise.
            SNR: The signal to noise ratio.

        Returns:
            A numpy array representing the linear combination of the signal and noise.
        """
        signal = np.array(signal)
        noise = np.array(noise)

        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        noise_scaled = noise * np.sqrt(SNR / ((signal_power / noise_power)+0.001))
        return signal + noise_scaled

    def draw_two_or_three_numbers_that_sum_to_one(self,number_of_numbers):
        """Draws two or three numbers that sum to 1."""
        if number_of_numbers == 2:
            first_number = random.uniform(0, 1)
            second_number = 1 - first_number
            return first_number, second_number
        elif number_of_numbers == 3:
            first_number = random.uniform(0, 1)
            second_number = random.uniform(0, 1 - first_number)
            third_number = 1 - first_number - second_number
            return first_number, second_number, third_number
        else:
            raise ValueError("Please enter 2 or 3.")



    def forward(self, signal, noise_type,SNR):
        # the choice is done outside, in the trainer.
        assert noise_type in ['MA','BW','EM','MA+BW','MA+EM','BW+EM', 'MA+BW+EM'], "Noise type must by one of ['MA','BW','EM','MA+BW','MA+EM','BW+EM', 'MA+BW+EM']."
        
        signal_length = len(signal)

        # create a noisy signal
        # If a signal noise was selected
        if noise_type in ['MA', 'BW', 'EM']:
            
            # draw a random index to start from
            noise_idx = np.random.randint(0,self.noise_signal_len-signal_length) 
            
            # get noise values
            noise = self.noise_signals[noise_type][noise_idx:(noise_idx+signal_length)]
        
        else:
            # break noise_type into seperated noises
            noise_types_ = noise_type.split('+')
            num_types = len(noise_types_)
            weights = self.draw_two_or_three_numbers_that_sum_to_one(num_types)
            indices = [np.random.randint(0,self.noise_signal_len-signal_length) for j in range(num_types)]

            # get noise values
            noises = []
            for noise_type, weight, indx in zip(noise_types_,weights, indices):
                # get noise values
                unscaled_noise=self.noise_signals[noise_type][indx:(indx+signal_length)]
                # print(type(unscaled_noise))
                unscaled_noise=np.array(unscaled_noise)
                # print(type(unscaled_noise))
                noise = np.array(unscaled_noise)*weight
                noises.append(noise)
                
            # create a combined noise array
            noise = sum(noises)

        # get corrupted signal
        corrupted_signal = self.create_linear_combination(signal, noise, SNR)

        return corrupted_signal
