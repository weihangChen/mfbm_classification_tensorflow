import numpy as np
from fbm1 import * 
from variables import *
from hurstactions import *
class Sender(object):
    def __init__(self, debug = False):
        self.debug = debug
        
        #alphabet a
        action_a_1 = HurstUpOrDown(0.6, 150, 600, "up")
        action_a_2 = HurstUpOrDown(0.6499999999999999, 350, 600, "down")
        hurst_actions_a = [action_a_1, action_a_2]
        hurst_config_a = HurstConfig(0.6, hurst_actions_a)
        #alphabet b
        action_b_1 = HurstUpOrDown(0.6499999999999999, 150, 600, "down")
        action_b_2 = HurstUpOrDown(0.6, 350, 600, "up")
        hurst_actions_b = [action_b_1,action_b_2]
        hurst_config_b = HurstConfig(0.6499999999999999, hurst_actions_b)

        #alphabet c
        hurst_config_c = HurstConfig(0.57, [None, None])

        self.letter_to_hurstconfig = dict({
            "a":hurst_config_a, "b":hurst_config_b, "c":hurst_config_c                          
        })  

    
    
    def gen_data_series_for_input(self, input_str):
        series = []
        for c in input_str:
            hurst_config = self.letter_to_hurstconfig[c]
            hurst = hurst_config.default_hurst
            hurst_actions = hurst_config.hurst_actions
            serie = fbm(n=seq_len-1, hurst=hurst, length=length, method=fbm_method, hurst_actions = hurst_actions)
            series.append(serie)

        return series