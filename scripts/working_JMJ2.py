import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import Statistics as st
import DataSets as ds
import FeatureGeneration as fg
import DataLoading as dl
import cleanResultantMerge as crm

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize


def GetDataSet(dense=True,sparse=True,specialdense=True):
    # LOAD DATA
    dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad('FeatureSet_A')
    
    if specialdense:
        missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
        engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls']        
        dense_df = dense_df[missingfieldindicators+engineeredfeatures]
    
    # NORMALIZE
    binary_col_selector = summary.distinct_count == 2
    nonbinary_col_selector = summary.distinct_count > 2
    binary_cols = dense_df.loc[:,binary_col_selector]
    nonbinary_cols = dense_df.loc[:,nonbinary_col_selector]
    normalized = pd.DataFrame(normalize(nonbinary_cols,norm='l2'),columns=nonbinary_cols.columns)
    dense_normalized = pd.concat((binary_cols,normalized),axis=1,ignore_index=True)        
    
    # COMBINE ALL FEATURES
    if dense and sparse:
        features = fg.CombineFeatures([dense_normalized],sparsefeatures)
        features = sp.sparse.csr_matrix(features) #required for efficient slicing
    elif dense:
        features = dense_normalized
    elif sparse:
        features = fg.CombineFeatures([],sparsefeatures)
        features = sp.sparse.csr_matrix(features) #required for efficient slicing
    
    # GET NUM DENSE & SPARSE (USED LATER IN COEF)
    numdense = dense_normalized.shape[1]
    numsparse = sparsefeatures[0].shape[1]
    numfeatures = numdense+numsparse
    
    selector_dense = np.arange(numfeatures) < numdense
    selector_sparse = selector_dense == False
    
    # TRAIN/TEST SLICING
    sel_bool_train = train == 1
    sel_bool_test = train == 0
    sel_ind_train = np.where(sel_bool_train)[0]
    sel_ind_test = np.where(sel_bool_test)[0]
    
    f_train = features[sel_ind_train]
    f_test = features[sel_ind_test]
    
    # N
    approved = 1-rejected
    y_train = np.array(approved[sel_bool_train]).astype(int)
    y_test = np.array(approved[sel_bool_test]).astype(int)
    
    return f_train,f_test,y_train,y_test

# CLASSIFIERS
'''
clf1 = MultinomialNB().fit(f_train, y_train)
probs = clf1.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
st.plotROC(fpr,tpr,roc_auc,"MultinomialNB")


clf2 = LogisticRegression(penalty='l1').fit(f_train, y_train)
probs = clf2.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
st.plotROC(fpr,tpr,roc_auc,"LogReg")


clf3 = SGDClassifier(penalty='l1').fit(f_train, y_train)
probs = clf3.decision_function(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
roc_auc = auc(fpr,tpr)
st.plotROC(fpr,tpr,roc_auc,"LogRegSGD")
'''

coef_dense = clf2.coef_[0][selector_dense]
coef_binary = st.coef_dataframe(
                    coef_dense[np.array(binary_col_selector)],
                    binary_cols.columns,
                    summary)
coef_numerical = st.coef_dataframe(
                    coef_dense[np.array(nonbinary_col_selector)],
                    nonbinary_cols.columns,
                    summary)
coef_sparse = st.coef_dataframe(
                    clf2.coef_[0][selector_sparse],
                    sparseheaders)
                    
roc = pd.DataFrame(np.hstack((np.reshape(tpr,(-1,1)),np.reshape(fpr,(-1,1)))),columns=['TPR','FPR'])

bool_approved = approved == 1
bool_rejected = approved == 0

ind_approved = np.where(bool_approved)
ind_rejected = np.where(bool_rejected)

sparse_approved = sparsefeatures[0][ind_approved]
sparse_rejected = sparsefeatures[0][ind_rejected]

avg_tfidf_approved = pd.DataFrame(
                            sparse_approved.mean(axis=0).reshape(-1,1),
                            index=sparseheaders,
                            columns=['avg_tfidf'])
avg_tfidf_rejected = pd.DataFrame(
                            sparse_rejected.mean(axis=0).reshape(-1,1),
                            index=sparseheaders,
                            columns=['avg_tfidf'])
                            
avg_tfidf_approved = avg_tfidf_approved.sort('avg_tfidf',ascending=False)
avg_tfidf_rejected = avg_tfidf_rejected.sort('avg_tfidf',ascending=False)

avg_tfidf_approved_top10 = avg_tfidf_approved.iloc[:10,:]         
avg_tfidf_rejected_top10 = avg_tfidf_rejected.iloc[:10,:]          

app_indices = avg_tfidf_approved.index
rej_indices = avg_tfidf_rejected.index

topwords = pd.concat(
                (pd.DataFrame(app_indices,columns=['Approved']),
                 pd.DataFrame(np.array(avg_tfidf_approved),columns=['AVGTFIDF_Approved']),
                 pd.DataFrame(rej_indices,columns=['Rejected']),
                 pd.DataFrame(np.array(avg_tfidf_rejected),columns=['AVGTFIDF_Rejected'])),
                axis = 1,
                ignore_index = False)

df = ds.pickleLoad('BalancedFull')
def ClosestRecord(threshold=0.01,label=1):
    # grabs records with label = 1 in test set with probability
    # closest to threshold
    myprobs = probs[:,1]
    probs_rejected = y_test == label
    diff_myprobs = np.abs(myprobs-threshold)
    minval = min(diff_myprobs[probs_rejected])
    ind_minval_rejected = np.logical_and(probs_rejected,diff_myprobs==minval)
    return df[sel_bool_test][ind_minval_rejected],dense_df[sel_bool_test][ind_minval_rejected]


#ds.pickleIt((coef_binary,coef_numerical,coef_sparse),'FeatureSetA_coef_summaries')


def GridSearch(data,params,classifier,classifier_name,paramname,probstype=1,clf_kwargs={}):
    f_train,f_test,y_train,y_test = data
    # C=1 is best
    def getROC(clf,probstype):
        if probstype == 1:
            probs = clf.predict_proba(f_test)
            fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
        else:
            probs = clf.decision_function(f_test)
            fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
        return fpr,tpr
    aucs = []
    mykwargs = clf_kwargs.copy()
    for c in params:
        mykwargs[paramname] = c
        clf = classifier(**mykwargs).fit(f_train, y_train)
        fpr,tpr = getROC(clf,probstype)
        roc_auc = auc(fpr,tpr)
        #cstr = '%0.2e'%c
        myplt = st.plotROC(fpr,tpr,roc_auc,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=False,
                    title='Grid Search: '+classifier_name+' ROC Curve')
        aucs.append(roc_auc)
    best = 0
    for i in range(len(params)):
        if aucs[i] > aucs[best]:
            best = i
    c = params[best]
    mykwargs[paramname] = c
    clf = classifier(**mykwargs).fit(f_train, y_train)
    fpr,tpr = getROC(clf,probstype)
    myplt = st.plotROC(fpr,tpr,roc_auc,
                    legendlabel='Best '+paramname+' = %0.2e' % c,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=True,
                    title='Grid Search: '+classifier_name+' ROC Curve')
    myplt.show()

    maxAUC = aucs[best]
    cs = params
    optC = params[best]
    
    plt.figure()
    maxauclabel = ("Max AUC = %0.2f, " %maxAUC)+paramname+(" =%s" %optC)
    plt.semilogx(cs,np.ones(len(cs))*maxAUC,'r',label=maxauclabel,linewidth=2,zorder=10)
    plt.semilogx(cs,aucs,zorder=1)
    plt.title('Grid Search: '+classifier_name+'AUC Scores')
    plt.xlabel(paramname)
    plt.ylabel('AUC Score')
    plt.legend(loc="lower right")
    #plt.legend(loc='lower left', bbox_to_anchor=(1, 0),
    #          ncol=1, fancybox=True, shadow=False)
    plt.show()
    
    return clf




def LogisticRegressionGridSearch(data):
    params = 10.0**np.arange(-1,2,0.25)  
    #params = [1]
    clf = GridSearch(data=data,
                     params=params,
                     classifier=LogisticRegression,
                     classifier_name="Logistic Regression",
                     paramname='C',
                     probstype=1,
                     clf_kwargs={'penalty':'l1'})

def MultinomialNaiveBayesGridSearch(data):
    params = 10.0**np.arange(-9,4,0.5)
    clf = GridSearch(data=data,
                     params=params,
                     classifier=MultinomialNB,
                     classifier_name="Multinomial Naive Bayes",
                     paramname='alpha',
                     probstype=1)

def SGDGridSearch(data):
    params = 10.0**np.arange(-14,2,1)
    clf = GridSearch(data=data,
                     params=params,
                     classifier=SGDClassifier,
                     classifier_name="SGD SVM",
                     paramname='alpha',
                     probstype=2,
                     clf_kwargs={'penalty':'l1'})


def runAllGridSearches():
    data = GetDataSet(dense=True,sparse=True,specialdense=True)
    MultinomialNaiveBayesGridSearch(data)
    SGDGridSearch(data)
    #LogisticRegressionGridSearch(data)
    
    
    
def SGDGridSearch_OLD():  
    # C=1 is best
    cs = 10.0**np.arange(-9,9,1)   
    aucs = []
    for c in cs:
        clf = SGDClassifier(penalty='l1',alpha=c).fit(f_train, y_train)
        probs = clf.decision_function(f_test)
        fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
        roc_auc = auc(fpr,tpr)
        cstr = '%0.2e'%c
        myplt = st.plotROC(fpr,tpr,roc_auc,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=False,
                    title='Grid Search - SGD Classifier ROC Curve')
        aucs.append(roc_auc)
    best = 0
    for i in range(len(cs)):
        if aucs[i] > aucs[best]:
            best = i
    c = cs[best]
    clf = SGDClassifier(penalty='l1',alpha=c).fit(f_train, y_train)
    probs = clf.decision_function(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
    myplt = st.plotROC(fpr,tpr,roc_auc,
                    legendlabel='Best C = %0.2e' % c,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=True,
                    title='Grid Search - SGD Classifier ROC Curve')
    myplt.show()
    return clf, aucs
    
def MultinomialNaiveBayesGridSearch_OLD():  
    # C=1 is best
    cs = 10.0**np.arange(-9,2,0.5)
    aucs = []
    for c in cs:
        clf = MultinomialNB(alpha=c).fit(f_train, y_train)
        probs = clf.predict_proba(f_test)
        fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
        roc_auc = auc(fpr,tpr)
        cstr = '%0.2e'%c
        myplt = st.plotROC(fpr,tpr,roc_auc,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=False,
                    title='Grid Search - Multinomial Naive Bayes ROC Curve')
        aucs.append(roc_auc)
    best = 0
    for i in range(len(cs)):
        if aucs[i] > aucs[best]:
            best = i
    c = cs[best]
    clf = MultinomialNB(alpha=c).fit(f_train, y_train)
    probs = clf.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    myplt = st.plotROC(fpr,tpr,roc_auc,
                    legendlabel='Best alpha = %0.2e' % c,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=True,
                    title='Grid Search - Multinomial Naive Bayes ROC Curve')
    myplt.show()
    return clf

def LogisticGridSearch_OLD():  
    # C=1 is best
    cs = 10.0**np.arange(-1,2,0.25)   
    aucs = []
    for c in cs:
        clf = LogisticRegression(penalty='l1',C=c).fit(f_train, y_train)
        probs = clf.predict_proba(f_test)
        fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
        roc_auc = auc(fpr,tpr)
        cstr = '%0.2e'%c
        myplt = st.plotROC(fpr,tpr,roc_auc,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=False,
                    title='Grid Search - Logistic Regression ROC Curve')
        aucs.append(roc_auc)
    best = 0
    for i in range(len(cs)):
        if aucs[i] > aucs[best]:
            best = i
    c = cs[best]
    clf = LogisticRegression(penalty='l1',C=c).fit(f_train, y_train)
    probs = clf.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    myplt = st.plotROC(fpr,tpr,roc_auc,
                    legendlabel='Best C = %0.2e' % c,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=True,
                    title='Grid Search - Logistic Regression ROC Curve')
    myplt.show()
    return clf