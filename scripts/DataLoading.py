# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:19:33 2014

@author: justinmaojones
"""

import pandas as pd
from utils import *
import os


def getDataFilePath(filename):
    mydir = os.path.dirname(os.path.realpath(__file__))
    pardir = os.path.join(mydir,"..")
    datadir = os.path.join(pardir,"data")
    return os.path.join(datadir,filename)


#################### BEGIN CHUNKING (DO NOT USE WHEN DIRECTLY READING TO DF)
# Generate the data frame by first creating creating an empty data frame
# and successively append each chunk to it.  To get the headers, grab the
# column names from a chunk of size 1.
@timethis
def LoadByChunking(filename,breakme=False,chunksize=10000,MaxChunks=5,*args,**kwargs):
	chunksize = 10000
	essays_labels = pd.read_csv(filename,iterator=True,chunksize=chunksize)

	firstchunk = essays_labels.get_chunk(1)
	headers = firstchunk.columns

	data_app = pd.DataFrame(columns=headers)
	data_rej = pd.DataFrame(columns=headers)

    # breakme is used to stop it from iterating through all the chunks when only
    # small batches sizes are wanted for testing.
	j = 0
	for chunk in essays_labels:
	    data_app = data_app.append(chunk[chunk.got_posted=='t'])
	    data_rej = data_rej.append(chunk[chunk.got_posted!='t'])
	    j=j+1
	    if breakme and j >= MaxChunks:
		break
	return data_app,data_rej,headers
################### END CHUNKING

################### READ DIRECTLY TO DATA FRAME, NO CHUNKING
@timethis
def ReadFullCSV(filename,*args,**kwargs):
    essays_labels = pd.read_csv(filename)
    headers = essays_labels.columns
    
    data_app = essays_labels[essays_labels.got_posted=='t']
    data_rej = essays_labels[essays_labels.got_posted!='t']
    
    return data_app,data_rej,headers
################### END READ
    
    
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
    
    # change labels to 1 and 0
    print "1 if Rejected, 0 if Approved"
    data_app = data_app.replace(to_replace={'got_posted':{'t':0,'f':1}})
    data_rej = data_rej.replace(to_replace={'got_posted':{'t':0,'f':1}})
    
    return data_app,data_rej

def getChunkedData(filename,breakme=True,MaxChunks=1):
    filepath = getDataFilePath(filename)
    data_app,data_rej,headers = LoadByChunking(filepath,breakme=True,MaxChunks=1)
    return ReviseDataLabels(data_app,data_rej)
    
def getFullData(filename):
    filepath = getDataFilePath(filename)
    data_app,data_rej,headers = ReadFullCSV(filepath)
    return ReviseDataLabels(data_app,data_rej)

  
    

#   Chunker reads 'filen' one chunk at a time,
#   with specified 'chunksize', then feeds
#   chunk' and 'chunk_id' to function 'func'
@timethis
def chunker(filen,func,chunksize=10000):
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
        

