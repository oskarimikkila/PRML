# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:30:23 2018

@author: oskar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:10:58 2018

@author: oskar
"""
    
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
import sklearn.cross_validation as crosval
from skimage import data
from sklearn import metrics

from sklearn.preprocessing import Normalizer


from sklearn.linear_model import LogisticRegression

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

    
    clf_list = [LogisticRegression(), SVC()]
    clf_name = ['LR', 'SVC']
    
    C_range = np.array([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
    
    best = 0
    bestC = -100
    bestPen = "4"
    
    for clf,name in zip(clf_list, clf_name):
        for C in C_range:
            for penalty in ["l1", "l2"]:
                clf.C = C
                clf.penalty = penalty
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.accuracy_score(y_test, y_pred)
                
                if score > best:
                    best = score
                    bestC = C
                    bestPen = penalty
                
                print("C: " + str(C) + " penalty: " + penalty + " score: " + str(score))
    print("\n Best ones: C: " + str(bestC) + " penalty: " + bestPen + " score: " + str(best))