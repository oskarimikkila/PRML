# -*- coding: utf-8 -*-
"""

Exercise 2:
    
Found here: http://www.cs.tut.fi/courses/SGN-41007/exercises/Exercise2.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.image import imread
        
def Ex2():    
    #x = np.array([7, 9, 2])
    #y = np.array([11.6, 14.8, 3.5])    
    
    # Load data
    x = np.load("x.npy")
    y = np.load("y.npy")
    
    X = np.column_stack([x, np.ones_like(x)])
    
    # Solve
    # y = ax + b
    a, b = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    print(a, b)
    # x range for plotting = [-11, 11[
    xn = np.arange(-11, 11, 0.1)
    
    plt.plot(x, y, 'ro')
    plt.plot(xn, a*xn + b, 'b-')
    
    plt.show()

def Ex3():
    
    # Load data without numpy
    filename = "locationData.csv"
    file = open(filename, 'r')

    data = []

    for line in file:
        l = line.strip().split(" ")
        
        # Convert  to float
        fl = [float(v) for v in l]
        data.append(fl)
        
    file.close
    # Cast to numpy array
    data = np.array(data)

    # Load data with numpy
    data2 = np.loadtxt(filename)
    
    # Confirm that same results
    print( (data == data2).any() )
    
def Ex4():
    # Load to dict
    mat = loadmat("twoClassData.mat")
    
    # Values from keys X and Y
    X = mat["X"]
    Y = mat["y"].ravel()
    
    # Plot class 0 = red, 1 = blue
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'ro')
    plt.plot(X[Y != 0, 0], X[Y != 0, 1], 'bo')
        
def Ex5():
    img = imread("uneven_illumination.jpg")
    plt.imshow(img, cmap='gray')        
    plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))

    X, Y = np.meshgrid(range(1300), range(1030))
    Z = img
    
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    
    H = np.column_stack([x**2,  y**2, x*y, x, y, np.ones_like(x)])
    
    theta = np.linalg.lstsq(H, z)[0]
    
    # Predict
    #z_pred = H @ theta
    z_pred = np.dot(H, theta)
    Z_pred = np.reshape(z_pred, X.shape)
    
    # Subtract & show
    S = Z - Z_pred
    plt.imshow(S, cmap = 'gray')
    plt.show()
    

if __name__ == "__main__":

    Ex2()
    Ex3()
    Ex4()
    Ex5()
    
    