import pandas as pd
from DataLoading import *
#from sqlalchemy import create_engine

           

def MergeLabelsAndEssays():
    filename = "essays_and_labels.csv"
    filepath_Labels = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Labs = pd.read_csv(filepath_Labels,iterator=True,chunksize=chunksize)          
    #df = Chunker.get_chunk(chunksize)
    cols_Labs = ['got_posted','_projectid']
    
    
    
    
    filename = "opendata_essays_2014_11_05.csv"
    filepath = getDataFilePath(filename)
    chunksize = 20000
    Chunker_Full = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
    cols_Full = ['_projectid','_teacher_acctid','title','short_description','need_statement','essay']
    #chunk = Chunker.get_chunk(chunksize)
    
    
    
    #merged = pd.merge(chunk,df,how='inner',on=["_projectid"])
    j=0
    useheaders = True
    for chunk in Chunker_Full:
        j=j+1
        print "chunk",j,"of 39"
        
        chunk = chunk[cols_Full]
        chunk._projectid = chunk._projectid.str.replace('"','')
        chunk._teacher_acctid = chunk._teacher_acctid.str.replace('"','')
        
        Chunker_Labs = pd.read_csv(filepath_Labels,iterator=True,chunksize=chunksize)
        for df in Chunker_Labs:
            df = df[cols_Labs]
            
            merged = pd.merge(chunk,df,how='inner',on=["_projectid"])
            with open(getDataFilePath("merged1.csv"),'a') as f:
                merged.to_csv(f,header=useheaders)
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
            


