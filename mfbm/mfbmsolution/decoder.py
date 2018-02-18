import tensorflow as tf
import numpy as np
from cnnmodels import *

class Decoder(object):
    def __init__(self, debug = False):
        self.debug = debug
        self.label_to_alphabet = dict({
            "0": "a", "1":"b", "2":"c"
        }) 
        #use cpu instead of gpu instead here, to avoid the short of memory warning
        with tf.device('/cpu:0'):
            self.inputs_, self.labels_, self.keep_prob_, self.logits, self.prediction = CNNModels().build_model()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_name)
    
    
    def classify(self, data):   
        feed = {self.inputs_: data, self.keep_prob_ : 1.0}        
        labels = self.sess.run(self.prediction, feed_dict=feed)
        alphabets = []
        for label in labels:
            alphabet = self.label_to_alphabet[str(label)]
            alphabets.append(alphabet)
        

        return alphabets
    
