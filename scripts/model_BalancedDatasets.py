import pandas as pd
import numpy as np

from Statistics import *
from TextProcessing import *
from DataSets import *
from FeatureGeneration import *

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc




df = pickleLoad("BalancedFull")
X = pickleLoad('BalancedFull_Essay_Vectorized')
Y = pickleLoad('BalancedFull_NeedStatement_Vectorized')
df.got_posted = df.got_posted.replace({'t':1,'f':0})

label = np.array(df.got_posted)

features = CombineFeatures([X,Y])
f_train,f_test,y_train,y_test = train_test_split(features,label,test_size=0.3)

y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)


clf = MultinomialNB().fit(f_train, y_train)
probs = clf.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
plotROC(fpr,tpr,roc_auc,"MultinomialNB")

clf = LogisticRegression().fit(f_train, y_train)
probs = clf.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
plotROC(fpr,tpr,roc_auc,"LogReg")

'''
clf = svm.LinearSVC().fit(f_train, y_train)
probs = clf.decision_function(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
roc_auc = auc(fpr,tpr)
plotROC(fpr,tpr,roc_auc,"LinSVM")
'''