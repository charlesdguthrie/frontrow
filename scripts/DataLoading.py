# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:19:33 2014

@author: justinmaojones
"""

import pandas as pd
from utils import *

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
