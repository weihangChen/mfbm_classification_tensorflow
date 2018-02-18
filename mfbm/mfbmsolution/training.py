from util import *
from cnnmodels import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from variables import *

X_train, labels_train = generate_training_data(count)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, 
                     stratify = labels_train, random_state = 123)

y_tr = one_hot(lab_tr,n_classes)
y_vld = one_hot(lab_vld,n_classes)

#add one dimension to data
X_tr = np.reshape(X_tr, [-1, seq_len, 1])
X_vld = np.reshape(X_vld, [-1, seq_len, 1])
validation_acc = []
validation_loss = []
train_acc = []
train_loss = []


model = CNNModels()
inputs_, labels_, keep_prob_, logits, prediction = model.build_model()
learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
# Cost function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gup_consumption
with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : keep_prob, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v,prediction_values = sess.run([cost, accuracy, prediction], feed_dict = feed)   
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1
    
    saver.save(sess,checkpoint_name)



# Plot training and test loss

if plot == True:
    t = np.arange(iteration-1)

    plt.figure(figsize = (6,6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()



    # Plot Accuracies
    plt.figure(figsize = (6,6))

    plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
    plt.xlabel("iteration")
    plt.ylabel("Accuray")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

#test_acc = []
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config = config) as sess:
    # Restore
    saver.restore(sess, checkpoint_name)
    
    feed = {inputs_: X_vld,
            labels_: y_vld,
            keep_prob_: 1}
        
    batch_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.6f}".format(batch_acc))
    tmp = sess.run(prediction, feed_dict = feed)
    print(tmp)