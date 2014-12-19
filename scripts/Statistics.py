# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:51:34 2014

@author: justinmaojones
"""
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from nltk import NaiveBayesClassifier
import numpy as np
import pandas as pd
from utils import *

def coef_dataframe(coef_array,indices,summary=[]):
    df = pd.DataFrame(
                np.vstack((coef_array,np.abs(coef_array))).T,
                index=indices,
                columns=['coef','coef_abs'])
    if len(summary) > 0:
        df = pd.concat((df,summary.loc[indices]),axis=1)
    df = df.sort(['coef_abs'],ascending=False)
    df = df.loc[:,df.columns != 'coef_abs']
    return df

def plotROC(fpr,tpr,roc_auc,legendlabel="",title="ROC Curves",figure=True,show=True,returnplt=False,showlegend=True):
    if figure:
        plt.figure()
    if len(legendlabel) > 0:
        plt.plot(fpr,tpr,label=legendlabel+" (AUC = %0.2f)" % roc_auc)
    else:
        plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    if showlegend:
        plt.legend(loc="lower right")
    if show:
        plt.show()
    if returnplt:
        return plt


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

def TrainTestSplit(data_app,data_rej,mytest_size=0.3):
    headers = data_app.columns
    
    print "Split: Train",mytest_size,"Test",1-mytest_size
    data_app_train, data_app_test = train_test_split(data_app,test_size=mytest_size)
    data_rej_train, data_rej_test = train_test_split(data_rej,test_size=mytest_size)
    
    data_app_train = pd.DataFrame(data_app_train,columns=headers)
    data_app_test = pd.DataFrame(data_app_test,columns=headers)
    data_rej_train = pd.DataFrame(data_rej_train,columns=headers)
    data_rej_test = pd.DataFrame(data_rej_test,columns=headers)
    
    return data_app_train,data_app_test,data_rej_train,data_rej_test

@timethis
def Classifier_NLTK_NaiveBayes(train,*args,**kwargs):
    return NaiveBayesClassifier.train(train)

