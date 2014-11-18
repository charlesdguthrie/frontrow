#util.py
import pandas as pd


'''
Contains utility functions for all of us to use.
'''

'''
Chunker reads 'filen' one chunk at a time,
with specified 'chunksize', then feeds
'chunk' and 'chunk_id' to function 'func'
'''
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