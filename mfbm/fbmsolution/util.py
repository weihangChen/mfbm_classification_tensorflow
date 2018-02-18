import pandas as pd 
import numpy as np
from variables import *
from fbm import *



def generate_training_data(count):    
    data = []
    labels = []
   
    for x in range(0,count):
        data_series = fbm(n=seq_len-1, hurst=hurst1, length=length, method=fbm_method)
        data.append(data_series)
        labels.append(1)
    
    for x in range(0,count):
        data_series = fbm(n=seq_len-1, hurst=hurst2, length=length, method=fbm_method)
        data.append(data_series)
        labels.append(2)

    for x in range(0,count):
        data_series = fbm(n=seq_len-1, hurst=hurst3, length=length, method=fbm_method)
        data.append(data_series)
        labels.append(3)
        
    return np.array(data), np.array(labels)


def one_hot(labels, n_class):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]
