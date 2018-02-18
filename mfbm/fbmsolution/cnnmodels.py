import tensorflow as tf
import numpy as np
from variables import *

class CNNModels:
    def __init__(self):
        pass



    def build_model(self):
        
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels_ = tf.placeholder(tf.float32, [None, 3], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        
        # (batch, 128, 9) --> (batch, 64, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                                    padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
        # (batch, 64, 18) --> (batch, 32, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                                    padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
        # (batch, 32, 36) --> (batch, 16, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                                    padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
        # (batch, 16, 72) --> (batch, 8, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                                    padding='same', activation = tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')


        
        # Flatten and add dropout
        #flat = tf.reshape(max_pool_4, (-1, 8*144))
        flat = tf.contrib.layers.flatten(max_pool_4)
        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
        # Predictions
        logits = tf.layers.dense(flat, 64)
        logits = tf.layers.dense(logits, 32)
        logits = tf.layers.dense(logits, 16)
        logits = tf.layers.dense(flat, n_classes)
    

        prediction = tf.argmax(logits,1)
        return inputs_, labels_, keep_prob_, logits, prediction

      