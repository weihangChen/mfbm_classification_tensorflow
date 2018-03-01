import numpy as np
from fbm import * 
from variables import *
from hurstactions import *
class Sender(object):
    def __init__(self, debug = False):
        self.debug = debug
      
        self.letter_to_hurstconfig = dict({
            "a":h1, "b":h2, "c":h3                          
        })  

    
    
    def gen_data_series_for_input(self, input_str):
        series = []
        for c in input_str:
            hurst_function = self.letter_to_hurstconfig[c]
            serie = mbm(n=seq_len-1, hurst=hurst_function, length=1, method=mfbm_method)
            series.append(serie)

        return series