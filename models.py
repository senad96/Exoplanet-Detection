import numpy as np
import tensorflow as tf
import pydot

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils.vis_utils import plot_model

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence



def FCN_model():
    
    
    '''
    model = tf.keras.Sequential()
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding = 'same' ,input_shape=(1,3197)))
   
    model.add(layers.BatchNormalization())
    
    model.add(layers.Activation('relu'))
    


    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, padding = 'same'))
   
    model.add(layers.BatchNormalization())
    
    model.add(layers.Activation('relu'))
    
    
    
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding = 'same'))
   
    model.add(layers.BatchNormalization())
    
    model.add(layers.Activation('relu'))
    
    
    
    
    
    model.add(layers.Conv1D(filters=1, kernel_size=3, strides=1, padding = 'same'))
   
    model.add(layers.BatchNormalization())
    
    model.add(layers.Activation('relu'))
    
    
    
    
    
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Softmax())
    
    
    return model
    
    '''
    
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
    
    
    
    '''
    input_seq = layers.Input(shape=(3197,1))

    # first convolution block
    x = layers.Conv1D(filters=64, kernel_size=8, padding = 'same',kernel_regularizer='l2')(input_seq)
    
    x = layers.BatchNormalization()(x)
    
    output_block1 = layers.Activation('relu')(x)
    
    
    
    #second block 
    x = layers.Conv1D(filters=120, kernel_size=8, padding = 'same',kernel_regularizer='l2')(output_block1)
    
    x = layers.BatchNormalization()(x)
    
    output_block2 = layers.Activation('relu')(x)
    
    
    
    # third block 
    x = layers.Conv1D(filters=120, kernel_size=3, strides=1, padding = 'same',kernel_regularizer='l2')(output_block2)
    
    x = layers.BatchNormalization()(x)
    
    output_block3 = layers.Activation('relu')(x)
    
    
    
    
    # fourth block 
    x = layers.Conv1D(filters=24, kernel_size=3, strides=1, padding = 'same',kernel_regularizer='l2')(output_block3)
    
    x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    output_block4 = layers.Flatten()(x)

    


    #fifth layer
    
    x = layers.Dense(12, activation = 'relu')(output_block4)
    
    output_block5 = layers.Dense(1, activation = 'sigmoid')(x)
    
    
    
    # last block
    
    x = layers.Dense(12, activation = 'relu')(output_block4)
    
    x = layers.Dense(4,activation = 'relu')(x)
    
    x = layers.Dense(1,activation = 'sigmoid')(x)
    
    
    #pred = layers.Activation('sigmoid')(x)
    
    model = tf.keras.Model(inputs = input_seq, outputs=output_block5)
    
    print(model.summary())
    '''
    

    return model
    