# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:47:06 2014

@author: Yasumasa
"""


def tokenize(raw):
    return nltk.word_tokenize(raw)
    
def text_obj(tokens):
    return nltk.Text(tokens)

def Stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [w.lower() for w in text if w.lower() not in stopwords]
    
def Symbols(text):
    import re
    return [w for w in text if re.search('[a-zA-Z]', w) and len(w) > 1]
    
def remove(raw):
    tokens = tokenize(raw)
    text = text_obj(tokens)
    removed = Stopwords(text)
    removed = Symbols(removed)
    return removed
    

    