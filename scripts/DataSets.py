from DataLoading import *
from TextProcessing import *

import pickle
import pandas as pd


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
    
def ImportPickleBalancedTruncated():
    df = pd.read_csv(getDataFilePath("Merge_2014_11_26.csv"))
    pickleIt(df, 'BalancedTruncated')
    print "Pickle Successful"

def PickleVectorized():
    df = pickleLoad('BalancedTruncated')
    X,_ = tfidf(df['essay'])
    Y,_ = tfidf(df['need_statement'])
    pickleIt(X, 'BalancedTruncated_Essay_Vectorized')
    pickleIt(Y, 'BalancedTruncated_NeedStatement_Vectorized')
