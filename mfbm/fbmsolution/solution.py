from decoder import *
from sender import *
from util import *
from variables import *
from fbm import *
import numpy as np
from cnnmodels import *

if __name__ == "__main__":
    sender = Sender()
    decoder = Decoder()
    while True:
        print('----------------')
        input_str = input('Enter your input:')
        try:
            input_str = input_str.strip()
            series = sender.gen_data_series_for_input(input_str)
            series = np.reshape(series, [-1, seq_len, 1])
            output = decoder.classify(series)
            print('classified content:{} '.format(output))
        except Exception as e:
            print("EXCEPTION: ")
            print(e)
      
            
        
