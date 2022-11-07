# -*- coding: utf-8 -*-
"""FNN.ipynb

"""

from keras.models import Sequential
from keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.utils import np_utils
#from imutils import paths
import numpy as np
import argparse
#import cv2
import os
from sklearn.metrics import matthews_corrcoef
import io
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from hypopt import GridSearch
import keras

def focal_loss(gamma=2., alpha=0.55):

    gamma = float(gamma)
    alpha = float(alpha)
    print(gamma,alpha)
    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto( device_count = {'GPU': len(tf.config.list_physical_devices('GPU'))} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

input_path = '/project/dmani2_uksr/dina_workplace/input-ori-BoT'
output_path = '/home/adi252/FNN-original/output'

X_train_file = 'train_norm.csv'
X_test_file = 'test_norm.csv'
X_valtest_file = 'valtest_norm.csv'


Y_train_file = 'y_train.csv'
Y_test_file = 'y_test.csv'
Y_valtest_file = 'y_valtest.csv'

train = pd.read_table(os.path.join(input_path, X_train_file), sep = ',',index_col = 0)
test = pd.read_table(os.path.join(input_path, X_test_file), sep = ',',index_col = 0)
valtest = pd.read_table(os.path.join(input_path, X_valtest_file), sep = ',',index_col = 0)


y_train = pd.read_table(os.path.join(input_path, Y_train_file),sep = ',', index_col = 0)
y_test = pd.read_table(os.path.join(input_path, Y_test_file),sep = ',', index_col = 0)
y_valtest = pd.read_table(os.path.join(input_path, Y_valtest_file),sep = ',', index_col = 0)
learning_rate = [0.01]

print(train.shape)
print(y_train.shape)
### FNN architecture
def create_model(learning_rate):
### FNN architecture
    model = Sequential()
    model.add(Dense(50, input_dim= 15, 
    	activation="relu"))
    model.add(Dense(30, activation="relu", kernel_initializer="uniform"))
    #model.add(Dense(30, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(5))
    model.add(Activation("softmax"))
    model.summary()
    adam = Adam(lr = learning_rate)
    # model.compile(loss="categorical_crossentropy", optimizer= adam,metrics=["accuracy"])
    model.compile(loss=focal_loss(alpha= 0.55), optimizer= adam,metrics=["accuracy"])	
    return model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 500,epochs = 30)
param_grids = dict(learning_rate = learning_rate)

dummy_y = np_utils.to_categorical(y_train)
y_valtest = np_utils.to_categorical(y_valtest)

# Build and fit the GridSearchCV

grid = GridSearch(model,param_grids, parallelize=False)
grid.fit(train,dummy_y, valtest, y_valtest)

model = grid
# testing test and test21
# dummy_test = np_utils.to_categorical(y_test)


### Classification report

# predict probabilities for test set
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

yhat_probs = model.predict(test)
print(yhat_probs)
# yhat_probs = np.argmax(yhat_probs,axis=0)
#yhat_classes = yhat_classes[:, 0]
print(yhat_probs)
y_pred = yhat_probs
y_test = np.array(y_test)
y_test = y_test.flatten()



## Classification report
report_DT = classification_report(y_test,y_pred,digits = 4)
mcc_ori = matthews_corrcoef(y_test, y_pred)
print("Test BoT",report_DT)
print("orininal test BoT",mcc_ori)
## saving predicted values
y_pred = pd.DataFrame(y_pred)


y_pred.to_csv(os.path.join(output_path,'test-pred-BoT.csv'), sep=',')




