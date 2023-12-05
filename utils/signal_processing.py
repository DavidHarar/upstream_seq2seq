import numpy as np
import scipy.signal as ss
from sklearn.preprocessing import MinMaxScaler


def resample(X, original_hz=500, desired_hz=200):
  """
  Resamples a signal to have the desired Hz.

  Args:
    signal: The signal to resample.
    original_hz: The original Hz of the signal.
    desired_hz: The desired Hz of the resampled signal.

  Returns:
    The resampled signals.
  """


  # num leads in signal (signal )
  batch_size, num_signals, signal_length = X.shape
  
  # Calculate the new sampling rate.
  new_length = int((desired_hz / original_hz)*signal_length)
  
  # Resample the signal.
  X_resampled = np.array([[ss.resample(X[i][j],new_length) for j in range(num_signals)] for i in range(batch_size)])
  return X_resampled

def normalize(X):
    batch_size, num_signals, signal_length = X.shape
    scaler = MinMaxScaler()
    
    return np.squeeze(np.array([[scaler.fit_transform(X[i][j].reshape((-1,1))) for j in range(num_signals)] for i in range(batch_size)]))

def sample(X1,X2, window=450, fix_start:int=None):
    """
    Sample a subsample of <length> from data
    """
    batch_size, num_signals, signal_length = X1.shape
    if fix_start:
       start = fix_start
    else:
        start = int(np.random.randint(0,signal_length-window-1,1))

    return X1[:,:,start:(start+window)], X2[:,:,start:(start+window)]

