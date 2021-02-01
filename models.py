import numpy as np
import tensorflow as tf
import pydot

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils.vis_utils import plot_model

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


#create the neural network
def FCN_model(len_seq):
    
    # len_seq = the size of the input sequences
    
    model = tf.keras.Sequential()
    
    #change the input shape if you have sequences less long
    model.add(layers.Conv1D(filters=256, kernel_size=8, activation='relu', input_shape=(len_seq,1)))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.BatchNormalization())
    
    
    model.add(layers.Conv1D(filters=340, kernel_size=6, activation='relu'))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.BatchNormalization())
    
    
    model.add(layers.Conv1D(filters=256, kernel_size=4, activation='relu'))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.BatchNormalization())
    
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    
    
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(12, activation='relu'))
    
    
    model.add(layers.Dense(8, activation = 'relu'))
    
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model



#create the SVC model
def SVC_model():
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    
    clf = GridSearchCV( SVC(), param_grid = tuned_parameters,scoring = 'recall')
    
    
    return clf