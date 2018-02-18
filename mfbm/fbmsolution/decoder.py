import tensorflow as tf
import numpy as np
from cnnmodels import *

class Decoder(object):
    def __init__(self, debug = False):
        self.debug = debug
        self.digit_length = 3
        self.digits_to_letter = dict({
            "000":"a", "001":"b","002":"c",
            "010":"d", "011":"e","012":"f",
            "020":"g", "021":"h","022":"i",
            "100":"j", "101":"k","102":"l",
            "110":"m", "111":"n","112":"o",
            "120":"p", "121":"q","122":"r",
            "200":"s", "201":"t","202":"u",
            "210":"v", "211":"w","212":"x",
            "220":"y", "221":"z","222":" "                           
        }) 
        #use cpu instead of gpu instead here, to avoid the short of memory warning
        with tf.device('/cpu:0'):
            self.inputs_, self.labels_, self.keep_prob_, self.logits, self.prediction = CNNModels().build_model()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_name)
    
    
    def classify(self, data):   
        feed = {self.inputs_: data, self.keep_prob_ : 1.0}        
        digits = self.sess.run(self.prediction, feed_dict=feed)
        output = ""
        digit_str = ""
        for index, x in enumerate(digits):
            digit_str = digit_str + str(x)
            if (index+1) % self.digit_length == 0:
                letter = self.digits_to_letter[digit_str]
                if self.debug == True:
                    print("classified digits is {} and it corresponds to letter {}".format(digit_str, letter))
                output += letter
                digit_str = ""
        return output
    
