import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,train_test_split
import matplotlib.pyplot as plt
import glob
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
import os  

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from tensorflow.keras.optimizers import *
def VGG(input_shape, categories):
    
    
    
    
    # inspired by VGG-16
    data = Input(shape = input_shape)
    
    X = Conv2D(6, (5,5), strides=(1,1), activation='relu', padding='same')(data)
    X = Conv2D(6, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X= Conv2D(16, (5,5), strides=(1,1), activation='relu', padding='same')(X)
    X= Conv2D(16, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X= Conv2D(22, (5,5), strides=(1,1), activation='relu', padding='same')(X)
    X= Conv2D(22, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    
    X = Flatten()(X)
    
    X = Dense(300)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(X)
    X = Activation('relu')(X)
    X = Dense(200)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(X)
    X = Activation('relu')(X)
    Y = Dense(2637, activation = 'softmax')(X)
    
    model = models.Model(inputs=data, outputs=Y, name='vggmodel')
    
    

    return model
# CNN = create_model((40,40,1),'categorical_crossentropy')
# CNN.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
#                  loss='categorical_crossentropy', metrics=['accuracy'])






