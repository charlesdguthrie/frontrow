'''
Functions for pickling and unpickling datasets
'''


from DataLoading import *
from TextProcessing import *
import FeatureGeneration as fg
import cleanResultantMerge as crm

import pickle
import pandas as pd
import numpy as np


def pickleIt(pyName, outputName):
    filepath = getDataFilePath(outputName)
    output = open(filepath+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()

def pickleLoad(inputName):
    filepath = getDataFilePath(inputName)
    pk1_file = open(filepath+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj
    
def ImportPickleBalancedFull(df):
    #df = pd.read_csv(getDataFilePath(infile))
    pickleIt(df, 'BalancedFull')
    print "Pickle Successful"

def PickleVectorized():
    df = pickleLoad('BalancedFull')
    print "vectorizing essays..."
    X,essay_words = tfidf(df['essay'])

    print "vectorizing need statements..."
    Y,need_words = tfidf(df['need_statement'])
    pickleIt(X, 'BalancedFull_Essay_Vectorized')
    pickleIt(Y, 'BalancedFull_NeedStatement_Vectorized')
    pickleIt(essay_words,'EssayWords')
    pickleIt(need_words,'NeedWords')

def FeatureSetA_Pickle():
    # LOAD DATA
    df = pickleLoad('BalancedFull')
    # FEATURE SET 1
    denseheaders,densefeatures = fg.getEssayFeatures(df)
    densefeatures[np.isnan(densefeatures)]=0
    # FEATURE SET 2
    df2 = fg.missingFieldIndicator(df)
    df2 = fg.dropFeatures(df2)
    df2 = fg.createDummies(df2)
    df2 = fg.replaceNansWithMean(df2)
    # ENTIRE DENSE FEATURE SET
    densefeatures2 = np.hstack((densefeatures,df2))
    dense_df_headers = denseheaders+list(df2.columns)
    dense_df = pd.DataFrame(densefeatures2,columns = dense_df_headers)
    # TAKE OUT LABEL
    rejected = dense_df.pop('rejected')
    train = dense_df.pop('train')
    # GET SUMMARY STATS
    summary = crm.getSummary(dense_df,rejected)
    dense_df = dense_df.loc[:,summary.distinct_count>1] #remove cols with only 1 distinct value
    summary = crm.getSummary(dense_df,rejected)
    summary = summary[summary.index != 'rejected']
    # SPARSE FEATURES
    essaywords = pickleLoad('EssayWords')
    essayvect = pickleLoad('BalancedFull_Essay_Vectorized')
    sparsefeatures = [essayvect]
    sparseheaders = sorted(essaywords.vocabulary_.keys(),key=lambda key: essaywords.vocabulary_[key])
    # PICKLE
    pickleIt((dense_df,train,rejected,summary,sparsefeatures,sparseheaders),"FeatureSet_A")


    
