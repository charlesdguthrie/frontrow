# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:46:28 2014

@author: justinmaojones
"""

from TextProcessing import *
import textmining
from utils import *

@timethis
def NLTKfeatures(df,lemmatize=False,*args,**kwargs):
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
            words = RemoveSymbolsAndSpecial(words)
            wordset = get_wordset(words)
            wordset = RemoveStopsSymbols(wordset)
            if lemmatize:
                wordset = lemmatizing(wordset)
            else:
                wordset = stemming(wordset)
            features = word_indicator(wordset)
            features_labels.append((features,label))
        except:
            print ">>>>>>>>>>ERROR"
            print "ROW",RowTuple[0]
            print row
            break
    return features_labels


def word_indicator(wordset, **kwargs):
    # Creates a dictionary of entries {word : True}
    # Note the returned dictionary does not include words not in the
    # string.  The NaiveBayesClassifier in NLTK only just requires {word : True}
    # and will create the full set of features behind the scenes.
    
    features = {}
    
    for w in wordset:
        features[w] = True
    return features
    



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