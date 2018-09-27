#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
import matplotlib.pyplot as plt
from ClassifyNB import classify
from AccuracyNB import NBAccuracy

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

def initialVisialization():
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train,features_test, labels_test)
    return accuracy

def submitClassify():

    clf = classify(features_train, labels_train)

    ### draw the decision boundary with the text points overlaid
    try:
        prettyPicture(clf, features_test, labels_test)
        output_image("test.png", "png", open("test.png", "rb").read())
    except NameError:
        pass

# INITIAL VIZ
initialVisialization()

# RUN LESSON 1
# submitClassify()

# RUN LESSON 2
# print("Accuracy = ", submitAccuracy())
