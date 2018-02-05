# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:10:58 2018

@author: oskar
"""
    
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import sklearn.cross_validation as crosval
from skimage import data
from sklearn import metrics

# settings for LBP
radius = 3
n_points = 8 * radius


def gaussian(x, mu, sigma):
    p = ( 1 / (np.sqrt(2*np.pi * pow(sigma, 2) )))* np.exp(- ( 1 /(2 * pow(sigma, 2)))*pow(x-mu, 2))
    return p


def log_gaussian(x, mu, sigma):
    
    p = np.log(1) \
        -np.log(np.sqrt(2*np.pi) *sigma ) \
        -( pow((x-mu), 2) /(2 * pow(sigma, 2)))

    return p

if __name__ == "__main__":
    
    
    mu = 0
    sigma = 1
    
    x = np.linspace(-5, 5)
    
    y_p = gaussian(x, mu, sigma)
    y_log = log_gaussian(x, mu, sigma)
    
    plt.plot(x, y_p, 'r', x, y_log, 'b')
    plt.show()
    
    log_gaussian(-2, mu, sigma)
    
    
    x = []
    y = []
    

    path = "C:/Users/oskar/Documents/Python/PRML/"
    for i in range(0, 101):
        
        if i < 10:
            imgstr = "00" + str(i)
        elif i < 100:
            imgstr = "0" + str(i)
        else:
            imgstr = str(i)
        
        x.append(data.load(path + "class1/" + imgstr+".jpg"))
        x.append(data.load(path + "class2/" + imgstr+".jpg"))
        
        y.append(1)
        y.append(2)
        
        
    hist = []
    
    X = []
    
    for img in x:
    
        X.append(local_binary_pattern(img, n_points, radius))

    for feat in X:
        hist.append(np.histogram(feat)[0])

    classifiers = [KNC(), LDA(), SVC()]
    
    x_train, x_test, y_train, y_test = crosval.train_test_split(
            hist, y, random_state=42, test_size = 0.25)

    for i in range(0, 3):
        classifiers[i].fit(x_train, y_train)
        res = classifiers[i].predict(x_test)
        print(metrics.accuracy_score(y_test, res))
        
    
        