# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:42:23 2014

@author: justinmaojones
"""

import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



def text_obj(tokens):
    return nltk.Text(tokens)

def Stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [w.lower() for w in text if w.lower() not in stopwords]
    
def Symbols(text):
    return [w for w in text if re.search('[a-zA-Z]', w) and len(w) > 1]

def RemoveSymbolsAndSpecial(words):
    for c in words:
        if not re.search('[a-zA-Z0-9_]',c):
            words = str.replace(words,c," ")
    return words

def RemoveStopsSymbols(tokens):
    text = text_obj(tokens)
    removed = Stopwords(text)
    #removed = Symbols(removed)  REDUNDANT WITH RemoveSymbolsAndSpecial
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
  

def lemmatizing(word_list):
  '''
  this function lemmatizes a list of words
  '''
  lemmatized = []
  lmt = WordNetLemmatizer()

  tagged = nltk.pos_tag(word_list)
  
  i = 0
  while i < len(tagged):
    wordnet_tagged = get_wordnet_pos(tagged[i][1])
    if wordnet_tagged == "":
      new_word = lmt.lemmatize(word_list[i])
    else:
      new_word = lmt.lemmatize(word_list[i],wordnet_tagged)
    lemmatized.append(new_word)
    i += 1

  return lemmatized

def get_wordnet_pos(tagged):
  '''
  this function converts the nltk pos tags to wordnet pos tags
  '''
  
  if tagged.startswith('J'):
    return wordnet.ADJ
  elif tagged.startswith('V'):
    return wordnet.VERB
  elif tagged.startswith('N'):
    return wordnet.NOUN
  elif tagged.startswith('R'):
    return wordnet.ADV
  else:
    return ""


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(df_column):
    
    # Fill missing data by "str" type values. 
    # Any input must be "str" type. "Nan" is float type. 
    df_column = df_column.fillna('')
    
    # Preprocessing. Each essay will be replaced by stemmed text data.
    k = 0
    for doc in df_column:
        
        words = RemoveSymbolsAndSpecial(doc)
        wordset = get_wordset(words)
        wordset = RemoveStopsSymbols(wordset)
        wordset = stemming(wordset)
        wordset = ' '.join(wordset)
        
        df_column[k] = wordset
        
        k = k + 1
        
    # Initialize vectorizer
    vectorizer = TfidfVectorizer()

    # Create a term document matrix
    x = vectorizer.fit_transform(df_column)

    return x, vectorizer
   
###################################################################################################   
    
from sklearn.feature_extraction.text import CountVectorizer

###Inputs and Outputs###
#  df_column   'pandas.core.series.Series' of 'str' type
#  method      'vect' option returns TFIDF matrix and vectorizer (default)
#              'count' option returns term frequency matrix and feature names  
def tfidf_2(df_column, method = 'vect'):
    
    # Fill missing data by "str" type values.
    # Any input must be "str" type. "Nan" is float type. 
    df_column = df_column.fillna('')
    
    k = 0
    
    # Preprocessing. Each essay will be replaced by stemmed text data.
    for doc in df_column:
        
        words = RemoveSymbolsAndSpecial(doc)
        wordset = get_wordset(words)
        wordset = RemoveStopsSymbols(wordset)
        wordset = stemming(wordset)
        wordset = ' '.join(wordset)
        
        df_column[k] = wordset
        
        k = k + 1
    
        
    # Initialize vectorizer
    if method == 'vect':
        vectorizer = TfidfVectorizer()
    elif method == 'count':
        vectorizer = CountVectorizer()
    else:
        print 'Error: Please choose a right method'
        return

    # Create a term document matrix
    x = vectorizer.fit_transform(df_column)
    
    if method == 'vect': 
        return x, vectorizer
    else:
        # Get feature names
        feature_names = vectorizer.get_feature_names()
        return x, feature_names
    
###Inputs and Outputs###  
#  x               'scipy.sparse.csc.csc_matrix' created by 'tfidf' method
#  feature_names   feature names created by 'tfidf' method. Data type should be 'list'
def sparse_to_DF(x, feature_names):
    
    # Create a Pandas DataFrame
    # x.A convert 'scipy.sparse.csc.csc_matrix' to 'numpy.ndarray'
    return pd.DataFrame(x.A, columns = feature_names)
