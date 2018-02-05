# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:23:27 2018

Exercise 3: 
    http://www.cs.tut.fi/courses/SGN-41007/exercises/Exercise3.pdf

@author: oskar
"""
import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as x
import sklearn.cross_validation as crosval
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def ex3():
    
    x = np.arange(0, 100, 1)
    
    f0 = 0.017
    sigma = 0.25
    
    for i in range(0, 10):
    
        w = np.sqrt(sigma) * np.random.randn(100)
        
        y = np.sin(x * f0 * 2 * np.pi) + w
    
        #plt.plot(x, y)
    
        scores = []
        frequencies = []
        
        for f in np.linspace(0, 0.5, 1000):
            
            n = np.arange(100)
            z = -2 * np.pi * 1j * f * n # <compute -2*pi*i*f*n. 
            e = np.exp(z)
            score = np.absolute(np.dot(y, e))# <compute abs of dot product of x and e>
            scores.append(score)
            frequencies.append(f)
        
        fHat = frequencies[np.argmax(scores)]
        
        print('%5.10f' % fHat)
    
    #print(frequencies)
    #print(scores)
    
def ex4():
    
    digits = x.load_digits()
    
    plt.gray()
    plt.imshow(digits.images[0])
    plt.show()
    
    print(digits.target[0])
    
    
    x_train, x_test, y_train, y_test = crosval.train_test_split(
            digits.data, digits.target, random_state=42, test_size = 0.25)

    clf = KNeighborsClassifier(n_neighbors = 6)

    clf.fit(x_train, y_train)
    
    y_result = clf.predict(x_test)
    
    acc = metrics.accuracy_score(y_test, y_result)
    
    print("Accuracy: "+ str(acc))


if __name__ == "__main__":
    
    ex4()
    
    
    
    
    
    
    """    
    test_xx = digits.data[0:1000]
    test_yy = digits.target[0:1000]
    
    result_yy = clf.predict(test_xx)
    
    plt.gray()
    for i in range(0, len(test_xx)):
        if result_yy[i] != test_yy[i]:
            plt.imshow(digits.images[i])
            plt.show()
            print("Pred: " + str(result_yy[i]))
            print("Real: " + str(test_yy[i]))"""
            