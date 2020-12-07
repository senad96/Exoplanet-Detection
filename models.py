import numpy as np
import tensorflow as tf
import pydot

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils.vis_utils import plot_model

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence


#create the neural network
def FCN_model():
    
    
    model = tf.keras.Sequential()
    
    
    
    model.add(layers.Conv1D(filters=128, kernel_size=12, activation='relu', input_shape=(3197,1)))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.BatchNormalization())
    
    
    model.add(layers.Conv1D(filters=256, kernel_size=5, activation='relu'))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.BatchNormalization())
    
    
    
    model.add(layers.Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(layers.MaxPool1D(strides=5))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    
    
    model.add(layers.Dense(36, activation='relu'))
    model.add(layers.Dropout(0.25))
    
    
    model.add(layers.Dense(12, activation='relu'))
    
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    print(model.summary())
    
    return model



#create the SVC model
def SVC_model():
    
    #model = 
    
    
    return #model