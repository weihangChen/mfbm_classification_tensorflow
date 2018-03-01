import numpy as np
from variables import *
from fbm import * 
import copy



def get_all_pathes():
    pathes = [[0.6],[0.6499999999999999]]
    nodes = [[0.6, 0.6499999999999999],[0.6, 0.6499999999999999]]
    for x in nodes:
        pathes = explore_next_step(pathes, x)

    #assert each combination is unique
    assert len(pathes) == 8

    return pathes

def explore_next_step(pathes, nodes_onestep_forward):
    pathes_onestep_forward = []
    for x in nodes_onestep_forward:
        pathes_copy = copy.deepcopy(pathes)
        for p in pathes_copy:
            p.append(x)
        pathes_onestep_forward.extend(pathes_copy)
    return pathes_onestep_forward

def get_next_step_hurst_action(step1_hurst, step2_hurst, step_index):
    start_index = 150
    if step_index == 2:
        start_index = 350
    if step1_hurst == step2_hurst:
        return None
    elif step2_hurst > step1_hurst:
        action = HurstUpOrDown(step1_hurst, start_index, seq_len, "up")
        return action
    elif step2_hurst < step1_hurst:
        action = HurstUpOrDown(step1_hurst, start_index, seq_len, "down")
        return action


def build_and_get_hurstfunctionconfigs():
    configs = []
    pathes = get_all_pathes()
    label_to_alphabet = dict({
        1: "a", 2:"b", 3:"c",
        4: "d", 5:"e", 6:"f",
        7: "g", 8:"h"
        })
    for x in pathes:
        n1 = x[0]
        n2 = x[1]
        n3 = x[2]
        hurst_actions = []
        action1 = get_next_step_hurst_action(n1, n2, 1)
        action2 = get_next_step_hurst_action(n2, n3, 2)
        hurst_actions.append(action1)
        hurst_actions.append(action2)
        hurst_config = HurstConfig(n1, hurst_actions)
        configs.append(hurst_config)

    for index, config in enumerate(configs):
        config.label = index+1
        config.alphabet = label_to_alphabet[index+1]
        
    #assert everything, see if it is correct
    for config in configs:
        assert config.alphabet is not None
        assert config.label is not None
        assert config.default_hurst is not None
        assert config.hurst_actions is not None and len(config.hurst_actions) == 2 


    #hard code the mapping of alphabet
    return configs


def generate_training_data1(count):    
    data = []
    labels = []
    configs = build_and_get_hurstfunctionconfigs()
    for config in configs:
        default_hurst = config.default_hurst
        hurst_actions = config.hurst_actions
        label = int(config.label)
        for x in range(0,count):       
            serie = fbm(seq_len-1, default_hurst, 1, 'daviesharte', hurst_actions)
            data.append(serie)
            labels.append(label)
    
    return np.array(data), np.array(labels)
   



def generate_training_data(count):    
    data = []
    labels = []


    for x in range(0,count):       
        serie = mbm(n=seq_len-1, hurst=h1, length=1, method=mfbm_method)
        data.append(serie)
        labels.append(1)
    
    for x in range(0,count):       
        serie = mbm(n=seq_len-1, hurst=h2, length=1, method=mfbm_method)
        data.append(serie)
        labels.append(2)

    for x in range(0,count):       
        serie = mbm(n=seq_len-1, hurst=h3, length=1, method=mfbm_method)
        data.append(serie)
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
