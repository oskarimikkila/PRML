# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:05:20 2018

@author: oskar
"""


import numpy as np
from skimage.feature import local_binary_pattern
import sklearn.cross_validation as crosval
from skimage import data
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# settings for LBP
radius = 3
n_points = 8 * radius

if __name__ == "__main__":

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

    norm = Normalizer()
    
    feat = norm.fit(feat)
    
    X_train, X_test, y_train, y_test = crosval.train_test_split(
            hist, y, random_state=42, test_size = 0.25)    
    
    classifiers = [RandomForestClassifier(n_estimators=100), 
                   ExtraTreesClassifier(n_estimators=100),
                   AdaBoostClassifier(n_estimators=100),
                   GradientBoostingClassifier(n_estimators=100)]
    
    names = ["RandomForestClassifier", "ExtraTreesClassifier",
             "AdaBoostClassifier", "GradientBoostingClassifier"]
    
    accuracies = []
    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        
    for a in accuracies:
        print(str(a))
    
    