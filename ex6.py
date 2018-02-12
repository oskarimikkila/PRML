# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:52:56 2018

@author: oskar
"""
import numpy as np

from skimage import data
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def load_images():
    
    path = "C:/Users/oskar/Documents/Python/PRML/GTSRB_subset_2/"
    x = []
    y = []
    
    for i in range(0, 450):
        
        if i < 10:
            imgstr = "00" + str(i)
        elif i < 100:
            imgstr = "0" + str(i)
        else:
            imgstr = str(i)
        
        x.append(data.load(path + "class1/" + imgstr+".jpg"))
        y.append(1)
        
    for i in range(210):     
        if i < 10:
            imgstr = "00" + str(i)
        elif i < 100:
            imgstr = "0" + str(i)
        else:
            imgstr = str(i)
        x.append(data.load(path + "class2/" + imgstr+".jpg"))
        y.append(2)
        
    return x, y


if __name__ == "__main__":
    N = 10 # Number of feature maps
    w, h = 5, 5 # Conv. window size
    
    X, y = load_images()
    
    X = (X - np.min(X)) / np.max(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 42)
    
    model = Sequential()
    
    model.add(Conv2D(N, (w, h),
                     input_shape=(64, 64, 3),
                     activation = 'relu',
                     padding = 'same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(N, (w, h),
                     activation = 'relu',
                     padding = 'same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(2, activation = 'sigmoid'))
    
    model.summary()
