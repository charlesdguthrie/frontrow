# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:51:34 2014

@author: justinmaojones
"""
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from nltk import NaiveBayesClassifier
import numpy as np
import pandas as pd


def plotROC(fpr,tpr,roc_auc):
    plt.figure()
    plt.plot(fpr,tpr,label="NaiveBayes (AUC = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
    

def getROC_NLTK(classifier,testset):
    # for NLTK classifiers ONLY
    # testset must be in format: [[{all words: True},label]]
    probs = []
    actual = []
    for item in testset:
        probs.append(classifier.prob_classify(item[0]).prob(1))
        actual.append(item[1])
    probs = np.array(probs)
    actual = np.array(actual)
    fpr,tpr,_ = roc_curve(y_true=actual,y_score=probs)
    roc_auc = auc(fpr,tpr)
    return fpr,tpr,roc_auc