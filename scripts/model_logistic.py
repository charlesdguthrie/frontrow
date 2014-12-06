'''
model_logistic.py
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

from cleanResultantMerge import *

rawdf = pickleLoad("BalancedFull")

def genFeatures(df):
    #get essay character count
    df['essay_len'] = df['essay'].str.len()
    
    for col in ['title','short_description','need_statement','essay']:
        #get null indicators for essays
        if len(df[col][pd.isnull(df[col])])>0:
            df[col+'_mv'] = np.where(pd.isnull(df[col]),1,0)
            
    return df

df = genFeatures(rawdf)

#drop columns that are not useful for logistic model
cols_to_drop = [
'_projectid', '_teacher_acctid', '_schoolid', 
'school_ncesid', 'school_latitude', 'school_longitude', 
'school_city', 'school_zip', 'school_district', 'school_county', 
'title', 'short_description', 'need_statement', 'essay', 
'school_zip_mv', 'school_ncesid_mv', 'school_district_mv', 'school_county_mv',
'fulfillment_labor_materials'
]

modeldf = df.drop(cols_to_drop, axis=1)


datasum = getSummary(modeldf)
#convert categorical variables to dummies
categorical = datasum[datasum.dtype == 'object'].index
for col in categorical:
    dummies = pd.get_dummies(modeldf[col])
    modeldf = pd.concat([modeldf,dummies],axis=1)
    modeldf = modeldf.drop(col, axis=1)

#replace nans with mean
for col in modeldf.columns:
    # if there are any nulls
    if len(modeldf[col][pd.isnull(modeldf[col])])>0:
        modeldf[col] = modeldf[col].replace(to_replace=np.nan, value=np.nanmean(modeldf[col]))

#save correlation matrix
modeldf.corr().to_csv('../data/corr_table.csv', index=True)


#run logistic regression
label = np.array(modeldf.rejected)
f_train,f_test,y_train,y_test = train_test_split(modeldf,label,test_size=0.3)
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

clf = LogisticRegression().fit(f_train, y_train)
probs = clf.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
plotROC(fpr,tpr,roc_auc,"LogReg")