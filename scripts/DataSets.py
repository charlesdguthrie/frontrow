'''
Functions for pickling and unpickling datasets
'''


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
    
def ImportPickleBalancedFull():
    df = pd.read_csv(getDataFilePath("Merge_2014_12_05.csv"))
    pickleIt(df, 'BalancedFull')
    print "Pickle Successful"

def PickleVectorized():
    df = pickleLoad('BalancedFull')
    print "vectorizing essays..."
    X,_ = tfidf(df['essay'])
    print "vectorizing need statements..."
    Y,_ = tfidf(df['need_statement'])
    pickleIt(X, 'BalancedFull_Essay_Vectorized')
    pickleIt(Y, 'BalancedFull_NeedStatement_Vectorized')
