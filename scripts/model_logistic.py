'''
model_logistic.py
'''

import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

from cleanResultantMerge import *
from DataSets import *
from FeatureGeneration import *

rawdf = pickleLoad("BalancedFull")

#record missing essay fields (very important)
df = missingFieldIndicator(rawdf)

#attach essay features
headers, essay_features = getEssayFeatures(df)
essayDf = pd.DataFrame(essay_features, columns=headers)
df = pd.concat([df,essayDf],axis=1)

#drop unnecessary features
df = dropFeatures(df)

df = createDummies(df)

df = replaceNansWithMean(df)


#add vectorized data


#split into train, test; and features vs outcome
train = df[df.train==1].drop('train', axis=1)
test = df[df.train==0].drop('train', axis=1)
f_train = train.drop('rejected',axis=1)
f_test = test.drop('rejected',axis=1)
y_train = train.rejected
y_test = test.rejected

y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

#run logistic regression
clf = LogisticRegression().fit(f_train, y_train)
probs = clf.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
plotROC(fpr,tpr,roc_auc,"LogReg")

#save coefficients
coeffs = pd.DataFrame(clf.coef_[0], index=f_train.columns)
datasum = getSummary(df)
datasum['coeffs'] = coeffs

