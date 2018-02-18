import numpy as np
from fbm import *
from variables import *
class Sender(object):
    def __init__(self, debug = False):
        self.debug = debug
        self.digit_to_hurst = dict({
            "0": hurst1, "1":hurst2, "2":hurst3
        }) 
        
        self.letter_to_digits = dict({
            "a":"000", "b":"001","c":"002",
            "d":"010", "e":"011","f":"012",
            "g":"020", "h":"021","i":"022",
            "j":"100", "k":"101","l":"102",
            "m":"110", "n":"111","o":"112",
            "p":"120", "q":"121","r":"122",
            "s":"200", "t":"201","u":"202",
            "v":"210", "w":"211","x":"212",
            "y":"220", "z":"221"," ":"222"                            
        })  

    
    
    def gen_data_series_for_input(self, input_str):
        series = []
        for c in input_str:
            digits = self.letter_to_digits[c]
            if self.debug == True:
                print('letter {} corresponds to digits {}'.format(c, digits))        
            for d in digits:
                hurst = self.digit_to_hurst[d]
                serie = fbm(n=seq_len-1, hurst=hurst, length=length, method=fbm_method)
                series.append(serie)

        return series