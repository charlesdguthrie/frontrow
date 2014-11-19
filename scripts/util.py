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
    



"""
begin{termdocumentmatrix}

Created on Tue Nov 18 20:46:48 2014

@author: Yasumasa

def termdocumentmatrix(df_column, preprocess = True)
@param   df_column    a column of Pandas DataFrame (i.e. full_essay, title etc.) 
@param   preprocess   (Optional) This parameter is 'True' by default.
@                     If 'True', use our preprocesing methods.
@                     If 'False', use the textmining's default methods. 
@
@   This method recieves pandas.core.series.Series of text data and outputs 
@   a term document matrix as pandas.core.frame.DataFrame. A user can choose 
@   two different preprocessig methods by setting the 'preprocess' parameter.
"""


import textmining


def termdocumentmatrix(df_column, preprocess = True):
    
    # Initialize a term document matrix
    matrix = textmining.TermDocumentMatrix()
    
    # Manipulate each essay
    for doc in df_column:            
        # Preprocessing 
        if preprocess == True:
            wordset = get_wordset(doc)
            trimmed = RemoveStopsSymbols(wordset)
            stemmed = stemming(trimmed)
            doc = ' '.join(stemmed)
       
        # Add documents to matrix
        matrix.add_doc(doc)
        
    # Create a list of lists    
    matrix_rows = []
    for row in matrix.rows(cutoff = 1):
        matrix_rows.append(row)
        
    # Convert to numpy array to store in DataFrame    
    matrix_array = np.array(matrix_rows[1:])
    matrix_terms = matrix_rows[0]
    df = pd.DataFrame(matrix_array, columns = matrix_terms)
    
    ## We can create a csv file also
    # matrix.write_csv('test_term_matrix.csv', cutoff=1)
    
    return df



# termdocumentmatrix() calls following functions
# Yasu will delete this later.
########################################################################

import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer


def text_obj(tokens):
    return nltk.Text(tokens)

def Stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [w.lower() for w in text if w.lower() not in stopwords]
    
def Symbols(text):
    return [w for w in text if re.search('[a-zA-Z]', w) and len(w) > 1]

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

def stemming(word_list):

  stemmer = PorterStemmer()
  
  stemmed_word_list = []

  for word in word_list:
    stemmed = stemmer.stem(word)
    stemmed_word_list.append(stemmed)

  return stemmed_word_list
  
########################################################################

'''
end{termdocumentmatrix}
'''
