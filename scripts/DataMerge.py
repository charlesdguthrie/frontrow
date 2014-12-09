import pandas as pd
from DataLoading import *
import os

#from sqlalchemy import create_engine

           

def MergeToFull(extractFileName, fullDf, outFileName, extractedCols):

    # extractFileName = "all_essays.csv"
    # fullFileName= "clean_labeled_project_data.csv"
    # outFileName = "data_and_essays.csv"
    # extractedCols = ['_projectid', 'title', 'short_description', 'need_statement', 'essay']

    #get file paths
    filepath = getDataFilePath(extractFileName)
    outFilePath = getDataFilePath(outFileName)

    #erase output file
    if (os.path.exists(outFilePath)):
        os.remove(outFilePath)

    chunksize = 50000
    
    #loop through chunks of essay data and merge
    useheaders = True
        
    chunker = pd.read_csv(filepath,iterator=True,chunksize=chunksize,dtype=unicode)
    for chunk in chunker:
        chunk = chunk[extractedCols]
        chunk._projectid = chunk._projectid.str.replace('"','')
        
        merged = pd.merge(fullDf,chunk,how='inner',on=["_projectid"])
        with open(outFilePath,'a') as f:
            merged.to_csv(f,header=useheaders, index=False)
        useheaders = False    

def MergeLabelsAndEssays(extractFileName, fullFileName, outFileName, extractedCols):

    # extractFileName = "all_essays.csv"
    # fullFileName= "clean_labeled_project_data.csv"
    # outFileName = "data_and_essays.csv"
    # extractedCols = ['_projectid', 'title', 'short_description', 'need_statement', 'essay']

    #get file paths
    filepath = getDataFilePath(extractFileName)
    fullFilePath = getDataFilePath(fullFileName)
    outFilePath = getDataFilePath(outFileName)

    #erase output file
    if (os.path.exists(outFilePath)):
        os.remove(outFilePath)

    chunksize = 50000
        
    #get columns from full file
    fullChunker = pd.read_csv(fullFilePath,iterator=True,chunksize=1)
    fullCols = fullChunker.get_chunk(1).columns

    #ok real chunker this time
    fullChunker = pd.read_csv(fullFilePath,iterator=True,chunksize=chunksize,dtype=unicode)
    
    #loop through chunks in metadata, then within each chunk, loop through chunks of essay data and merge
    j=0
    useheaders = True
    for chunk in fullChunker:
        j=j+1
        print "chunk",j,"of approx 2"
        
        #make sure to not exceed starting columns
        chunk = chunk[fullCols]
        
        chunker = pd.read_csv(filepath_essays,iterator=True,chunksize=chunksize,dtype=unicode)
        for df in chunker:
            df = df[extractedCols]
            df._projectid = df._projectid.str.replace('"','')
            
            merged = pd.merge(chunk,df,how='inner',on=["_projectid"])
            with open(outFilePath,'a') as f:
                merged.to_csv(f,header=useheaders, index=False)
            useheaders = False     


def CountFullEssaysDataSet():   
    filename = "opendata_essays_2014_11_05.csv"
    filepath = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Full = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    N_Full = 0
    for chunk in Chunker_Full:
        N_Full += len(chunk)
    print "Full essays data set:",N_Full,"records"
    
    
def CountEssaysAndLabelsCSV(): 
    filename = "essays_and_labels.csv"
    filepath = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Labs = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    N = 0
    app = 0
    rej = 0
    for chunk in Chunker_Labs:
        N += len(chunk)
        app += np.sum(chunk.got_posted=='t')
        rej += np.sum(chunk.got_posted=='f')
    print "Total:",N
    print "Approved:",app
    print "Rejected:",rej
    
    
def SliceOutProjectIDsFromMerged():    
    filename = "merged1.csv"
    filepath = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Merged = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    N = 0
    neg = 0
    pos = 0
    useheaders = True
    for chunk in Chunker_Merged:
        N += len(chunk)
        neg += np.sum(chunk.got_posted=='f')
        pos += np.sum(chunk.got_posted=='t')
        with open(getDataFilePath("merged_projectids.csv"),'a') as f:
            projectid = chunk[['got_posted','_projectid']]        
            projectid.to_csv(f,header=useheaders)
            useheaders=False
        
        
    

def SliceOutProjectIDsFromEssaysAndLabelsCSV():
    filename = "essays_and_labels.csv"
    filepath = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Labelled = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    N = 0
    neg = 0
    pos = 0
    useheaders = True
    for chunk in Chunker_Labelled:
        N += len(chunk)
        neg += np.sum(chunk.got_posted=='f')
        pos += np.sum(chunk.got_posted=='t')
        with open(getDataFilePath("labelled_projectids.csv"),'a') as f:
            projectid = chunk[['got_posted','_projectid']]        
            projectid.to_csv(f,header=useheaders)
            useheaders=False


def CompareAgainstProjectIDs():
    filename = "merged_projectids.csv"
    filepath = getDataFilePath(filename)
    merged_projectids = pd.read_csv(filepath)
    
    filename = "labelled_projectids.csv"
    filepath = getDataFilePath(filename)
    labelled_projectids = pd.read_csv(filepath)
    
    df = pd.merge(labelled_projectids,merged_projectids,how='inner',on=["_projectid"])
    return df








#
##
###
####
#####
######
#######
########
#########










#*****************************************************
#  IGNORE ALL HDF5 FUNCTIONS, BUT DONT DELETE

def loadLabelledHDF5():
    ### DONT USE!!!!!!!!!!!!
    filename = "essays_and_labels.csv"
    filepath = getDataFilePath(filename)
    chunksize = 10000
    Chunker = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    
    store = pd.HDFStore(getDataFilePath('mystore.h5'))
    store.close()
    
    with pd.get_store(getDataFilePath('mystore.h5')) as store:
        try:
            del store['labelled']
        except:
            pass
        for chunk in Chunker:
            chunk.replace(to_replace={'got_posted':{'t':1,'f':0}})
            store.append('labelled',chunk[['got_posted','_projectid']])
        print store

def loadFullNoLabelsHDF5():
    ### DONT USE!!!!!!!!!!!!
    with pd.get_store(getDataFilePath('mystore.h5')) as store:
        filename = "opendata_essays_2014_11_05.csv"
        filepath = getDataFilePath(filename)
        chunksize = 10000
        Chunker = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
        
        try:
            del store['full_no_labels']
        except:
            pass
        
        cols = ['_projectid','_teacher_acctid','title','short_description','need_statement','essay']
        chunk = Chunker.get_chunk(chunksize)
        chunk = chunk[cols]
        chunk._projectid = chunk._projectid.str.replace('"','')
        chunk._teacher_acctid = chunk._teacher_acctid.str.replace('"','')
        store.append('full_no_labels',chunk,min_itemsize=20000)
        
        for chunk in Chunker:
            chunk = chunk[cols]
            chunk._projectid = chunk._projectid.str.replace('"','')
            chunk._teacher_acctid = chunk._teacher_acctid.str.replace('"','')
            store.append('full_no_labels',chunk)
        print store
        
def HDF5merge():
    ### DONT USE!!!!!!!!!!!!
    with pd.get_store(getDataFilePath('mystore.h5')) as store:
        filename = "opendata_essays_2014_11_05.csv"
        filepath = getDataFilePath(filename)
        chunksize = 10000
        Chunker = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
        
        try:
            del store['merged']
        except:
            pass
        
        for chunk in Chunker:
            chunk._projectid = chunk._projectid.str.replace('"','')
            for df in store.select('labelled', chunksize=chunksize):
                merged = pd.merge(chunk,df,how='inner',on=["_projectid"])
                store.append('merged',merged,min_itemsize=20000)

    with pd.get_store(getDataFilePath('mystore.h5')) as store:          
        print store
            


