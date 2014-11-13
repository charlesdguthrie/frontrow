import pandas as pd
import sklearn as sk
import sklearn.linear_model as lm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import re
import nltk
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.cross_validation import train_test_split

file_essays = "full_labeled_essays_1000.csv"
file_essays_labels = "essays_and_labels.csv"
file_resources = "opendata_resources.csv"

chunksize = 1000
essays = pd.read_csv(file_essays,iterator=True,chunksize=chunksize)
essays_labels = pd.read_csv(file_essays_labels,iterator=True,chunksize=chunksize)
resources = pd.read_csv(file_resources,iterator=True,chunksize=chunksize)

nT = 0
nF = 0
for chunk in essays_labels:
    m,n = chunk.shape
    numtrue = np.sum(chunk.got_posted=="t")
    nT += numtrue
    nF += m-numtrue

def process_label(label):
    if label == "t":
        return True
    else:
        return False

def generate_features(df,features_labels=[]):
    # note, this will assign the same label input to all features
    
    m,n = df.shape
    for i in range(m):
        row = df.irow(i)
        title = str(row.title)
        essay = str(row.full_essay)
        needs = str(row.need_statement)
        label = process_label(row.got_posted)
        words = title + " " + essay + " " + needs
        features = word_indicator(words)
        features_labels.append((features,label))
    return features_labels
        
        
    

def get_wordset(string, stopwords=[], strip_html=True):
    # Create a set of all tokenized words in string, and remove stopwords.
    # Returns in list format
    tokenized = wordpunct_tokenize(string.lower())
    tokenset = set(tokenized)
    tokenset = tokenset.difference(stopwords)
    tokensetlist = [t for t in tokenset]
    return tokensetlist


def word_indicator(string, **kwargs):
    # Creates a dictionary of entries {word : True}
    # Note the returned dictionary does not include words not in the
    # string.  The NaiveBayesClassifier in NLTK only just requires {word : True}
    # and will create the full set of features behind the scenes.
    
    features = {}
    
    words = get_wordset(string, **kwargs)
    for w in words:
        features[w] = True
    return features



test_size = 0.3

features_labels = []

#for each chunk in essays:

#chunk = essays.get_chunk(chunksize)
m,n = chunk.shape
data_train, data_test = train_test_split(chunk,test_size=test_size)

data_train = pd.DataFrame(data_train,columns=headers)
data_test = pd.DataFrame(data_test,columns=headers)

headers = chunk.columns
train_set = generate_features(data_train)
test_set = generate_features(data_test)

classifier = NaiveBayesClassifier.train(train_set)

print ('Test accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_set)))
