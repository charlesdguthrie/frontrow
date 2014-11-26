# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:19:33 2014

@author: justinmaojones
"""

import pandas as pd
import numpy as np
from utils import *
from sklearn.cross_validation import train_test_split
import os


def getDataFilePath(filename):
    mydir = os.path.dirname(os.path.realpath(__file__))
    pardir = os.path.join(mydir,"..")
    datadir = os.path.join(pardir,"data")
    return os.path.join(datadir,filename)

def GetBalancedDataSet(filename):
    data_app_raw,data_rej = getChunkedData(filename,breakme=False)
    
    n_app = len(data_app_raw)
    n_rej = len(data_rej)
    ratio = n_rej*1.0/n_app
    data_app_train, data_app_ignore = train_test_split(data_app_raw,test_size=1-ratio)
    data_app = pd.DataFrame(data_app_train,columns=data_app_raw.columns)
    
    print "**********************************"
    print "BALANCED DATA:"
    print "approved data set:"
    print "   approved =",np.sum(data_app.got_posted==1)
    print "   rejected =",np.sum(data_app.got_posted==0)
    print "rejected data set:"
    print "   approved =",np.sum(data_rej.got_posted==1)
    print "   rejected =",np.sum(data_rej.got_posted==0)
    print "**********************************"
    return data_app, data_rej

def ReviseDataLabels(data_app,data_rej):
    print "Total Approved:",data_app.shape[0]
    print "Total Rejected:",data_rej.shape[0]
    
    # <<<<< MAY NEED TO BE REVISED IN FUTURE >>>>>
    # for now, just get rid of rows with missing labels.  there aren't that
    # many anyway.
    print "Total Approved with Missing Labels:",sum(data_app.got_posted.isnull())
    print "Total Rejected with Missing Labels:",sum(data_rej.got_posted.isnull())

    data_app = data_app[data_app.got_posted.isnull()==False]
    data_rej = data_rej[data_rej.got_posted.isnull()==False]
    print "Records with missing labels REMOVED"    
    
    # change labels to 1 and 0
    print "1 if Rejected, 0 if Approved"
    data_app = data_app.replace(to_replace={'got_posted':{'t':1,'f':0}})
    data_rej = data_rej.replace(to_replace={'got_posted':{'t':1,'f':0}})
    
    return data_app,data_rej

#################### BEGIN CHUNKING (DO NOT USE WHEN DIRECTLY READING TO DF)
# Generate the data frame by first creating creating an empty data frame
# and successively append each chunk to it.  To get the headers, grab the
# column names from a chunk of size 1.
@timethis
def LoadByChunking(filename,label_approved='t',breakme=True,chunksize=10000,MaxChunks=5,*args,**kwargs):
    essays_labels = pd.read_csv(filename,iterator=True,chunksize=chunksize)

    firstchunk = essays_labels.get_chunk(1)
    headers = firstchunk.columns

    data_app = pd.DataFrame(columns=headers)
    data_rej = pd.DataFrame(columns=headers)
    print "label_approved =",label_approved
    # breakme is used to stop it from iterating through all the chunks when only
    # small batches sizes are wanted for testing.
    j = 0
    for chunk in essays_labels:
        data_app = data_app.append(chunk[chunk.got_posted==label_approved],ignore_index=True)
        data_rej = data_rej.append(chunk[chunk.got_posted!=label_approved],ignore_index=True)
        j=j+1
        if breakme and j >= MaxChunks:
            print "delivered",j,"chunks of totaling",j*chunksize,"records"
            break
    print "approved:",len(data_app)
    print "rejected:",len(data_rej)
    return data_app,data_rej,headers

def getChunkedData(filename,breakme=True,label_approved='t',MaxChunks=1):
    filepath = getDataFilePath(filename)
    data_app,data_rej,headers = LoadByChunking(filepath,breakme=breakme,label_approved=label_approved,MaxChunks=MaxChunks)
    return ReviseDataLabels(data_app,data_rej)
################### END CHUNKING    

################### READ DIRECTLY TO DATA FRAME, NO CHUNKING
@timethis
def ReadFullCSV(filename,*args,**kwargs):
    essays_labels = pd.read_csv(filename)
    headers = essays_labels.columns

    data_app = essays_labels[essays_labels.got_posted==0]
    data_rej = essays_labels[essays_labels.got_posted!=0]
    
    return data_app,data_rej,headers
################### END READ
    
def getFullData(filename):
    filepath = getDataFilePath(filename)
    data_app,data_rej,headers = ReadFullCSV(filepath)
    return ReviseDataLabels(data_app,data_rej)

  
    

#   Chunker reads 'filen' one chunk at a time,
#   with specified 'chunksize', then feeds
#   chunk' and 'chunk_id' to function 'func'
def chunker(chunksize,filen,func):
    #read file one chunk at a time
    with open(filen, 'rb') as inf:
        reader = pd.read_csv(inf, delimiter=',', quotechar='"',iterator=True, chunksize=chunksize)

        #Read and process chunks
        for chunk_id,chunk in enumerate(reader):
            #print chunk.shape[1]
            func(chunk,chunk_id)


def write_chunk(outpath,outchunk):
    with open(outpath, 'a') as f:
        outchunk.to_csv(f, header=False, index=False)
        

