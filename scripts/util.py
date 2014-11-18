#util.py
import pandas as pd
import re
import nltk
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.cross_validation import train_test_split
from stemming import stemming
import os
import time

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

def RunItTimeIt(func, args=[], returnarg=False, msg=""):
    # This just runs the function func with args and prints a message
    # about how long it took.  You can attach a little message that
    # goes with it.  For example, "Function X completed in [5 seconds]".
    # If no return output required, set returnarg to false, otherwise
    # it will return the output of func.
    #
    # NOTE: args must be a list
    start = time.time()
    if returnarg == True:
        if len(args)==0:
            outargs = func()
        else:
            outargs = func(*args)
    else:
        if len(args)==0:
            func()
        else:
            func(*args)
    end = time.time()
    print msg, round(end-start,2),"seconds"
    if returnarg == True:
        return outargs

def write_chunk(outpath,outchunk):
    with open(outpath, 'a') as f:
        outchunk.to_csv(f, header=False, index=False)

def generate_features(df):
    # note, this will assign the same label input to all features
    
    features_labels=[]
    m,n = df.shape
    for RowTuple in df.iterrows():
        try:
            row = RowTuple[1]
            title = str(row["title"])
            essay = str(row["essay"])
            needs = str(row["need_statement"])
            label = row["got_posted"]
            words = title + " " + essay + " " + needs
            words = RemoveSpecialUnicode(words)
            wordset = get_wordset(words)
            trimmed = RemoveStopsSymbols(wordset)
            stemmed = stemming(trimmed)
            features = word_indicator(stemmed)
            features_labels.append((features,label))
        except:
            print ">>>>>>>>>>ERROR"
            print "ROW",RowTuple[0]
        print row
        break
    return features_labels

        
def text_obj(tokens):
    return nltk.Text(tokens)

def Stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [w.lower() for w in text if w.lower() not in stopwords]
    
def Symbols(text):
    return [w for w in text if re.search('[a-zA-Z]', w) and len(w) > 1]

def RemoveSpecialUnicode(words):
    for c in words:
        if not re.search('[a-zA-Z0-9_]',c):
            words = str.replace(words,c," ")
    return words
    
def RemoveStopsSymbols(tokens):
    text = text_obj(tokens)
    removed = Stopwords(text)
    removed = Symbols(removed)
    return removed    

def get_wordset(string, stopwords=[], strip_html=True):
    # Create a set of all tokenized words in string, and remove stopwords.
    # Returns in list format
    tokenized = wordpunct_tokenize(string.lower())
    tokenset = set(tokenized)
    tokenset = tokenset.difference(stopwords)
    tokensetlist = [t for t in tokenset]
    return tokensetlist


def word_indicator(wordset, **kwargs):
    # Creates a dictionary of entries {word : True}
    # Note the returned dictionary does not include words not in the
    # string.  The NaiveBayesClassifier in NLTK only just requires {word : True}
    # and will create the full set of features behind the scenes.
    
    features = {}
    
    for w in wordset:
        features[w] = True
    return features