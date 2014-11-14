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
from stemming import stemming

file_essays = "full_labeled_essays_1000.csv"
file_essays_labels = "essays_and_labels.csv"
file_resources = "opendata_resources.csv"

chunksize = 10000
essays = pd.read_csv(file_essays,iterator=True,chunksize=chunksize)
essays_labels = pd.read_csv(file_essays_labels,iterator=True,chunksize=chunksize)
resources = pd.read_csv(file_resources,iterator=True,chunksize=chunksize)

'''

nT = 0
nF = 0

for chunk in essays_labels:
    m,n = chunk.shape
    numtrue = np.sum(chunk.got_posted=="t")
    nT += numtrue
    nF += m-numtrue
'''

def process_label(label):
    if label == "t":
        return True
    else:
        return False

def generate_features(df):
    # note, this will assign the same label input to all features
    
    features_labels=[]
    m,n = df.shape
    for i in range(m):
        row = df.irow(i)
        title = str(row.title)
        essay = str(row.essay)
        needs = str(row.need_statement)
        label = row.got_posted
        words = title + " " + essay + " " + needs
        wordset = get_wordset(words)
        trimmed = RemoveStopsSymbols(wordset)
        stemmed = stemming(trimmed)
        features = word_indicator(stemmed)
        features_labels.append((features,label))
    return features_labels
        
        
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


def word_indicator(wordset, **kwargs):
    # Creates a dictionary of entries {word : True}
    # Note the returned dictionary does not include words not in the
    # string.  The NaiveBayesClassifier in NLTK only just requires {word : True}
    # and will create the full set of features behind the scenes.
    
    features = {}
    
    for w in wordset:
        features[w] = True
    return features



mytest_size = 0.3

features_labels = []

#for each chunk in essays:

#chunk = essays.get_chunk(chunksize)

headers = essays_labels.get_chunk(1).columns

data_app = pd.DataFrame(columns=headers)
data_rej = pd.DataFrame(columns=headers)

for chunk in essays_labels:
    data_app = data_app.append(chunk[chunk.got_posted=='t'])
    data_rej = data_rej.append(chunk[chunk.got_posted!='t'])

data_app = data_app[data_app.got_posted.isnull()==False]
data_rej = data_rej[data_rej.got_posted.isnull()==False]

headers = data_app.columns

data_app = data_app.iloc[-1000:,:]
data_rej = data_rej.iloc[-1000:,:]

data_app = data_app.replace(to_replace={'got_posted':{'t':1,'f':0}})
data_rej = data_rej.replace(to_replace={'got_posted':{'t':1,'f':0}})

data_app_train, data_app_test = train_test_split(data_app,test_size=mytest_size)
data_rej_train, data_rej_test = train_test_split(data_rej,test_size=mytest_size)

data_app_train = pd.DataFrame(data_app_train,columns=headers)
data_app_test = pd.DataFrame(data_app_test,columns=headers)
data_rej_train = pd.DataFrame(data_rej_train,columns=headers)
data_rej_test = pd.DataFrame(data_rej_test,columns=headers)



train_app = generate_features(data_app_train)
print "finished generating features: training approved"
test_app = generate_features(data_app_test)
print "finished generating features: test approved"
train_rej = generate_features(data_rej_train)
print "finished generating features: training rejected"
test_rej = generate_features(data_rej_test)
print "finished generating features: test rejected"

train = train_app + train_rej
test = test_app + test_rej

classifier = NaiveBayesClassifier.train(train)

print ('Test approved accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_app)))
print ('Test rejected accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_rej)))


nT=0
nF=0

for i in test:
    if i[1]==True:
        nT+=1
    else:
        nF+=1
print nT
print nF

