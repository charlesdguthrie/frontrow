import pandas as pd
import sklearn as sk
#import sklearn.linear_model as lm
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt
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
import os
import time

mydir = os.path.dirname(os.path.realpath(__file__))
pardir = os.path.join(mydir,"..")
datadir = os.path.join(pardir,"data")

file_essays = os.path.join(datadir,"full_labeled_essays_1000.csv")
file_essays_labels = os.path.join(datadir,"full_labeled_essays.csv")
file_essays_labels2 = os.path.join(datadir,"essays_and_labels.csv")
file_resources = os.path.join(datadir,"opendata_resources.csv")





def process_label(label):
    if label == "t":
        return True
    else:
        return False

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



mytest_size = 0.3

features_labels = []

'''
################### READ DIRECTLY TO DATA FRAME, NO CHUNKING
essays_labels = pd.read_csv(file_essays_labels2)
headers = essays_labels.columns

data_app = essays_labels[essays_labels.got_posted=='t']
data_rej = essays_labels[essays_labels.got_posted!='t']
################### END READ
'''


#################### BEGIN CHUNKING (DO NOT USE WHEN DIRECTLY READING TO DF)
# Generate the data frame by first creating creating an empty data frame
# and successively append each chunk to it.  To get the headers, grab the
# column names from a chunk of size 1.

def LoadByChunking(filename,breakme=False,chunksize=10000,MaxChunks=5):
	chunksize = 10000
	essays_labels = pd.read_csv(file_essays_labels2,iterator=True,chunksize=chunksize)

	firstchunk = essays_labels.get_chunk(1)
	headers = firstchunk.columns

	data_app = pd.DataFrame(columns=headers)
	data_rej = pd.DataFrame(columns=headers)

    # breakme is used to stop it from iterating through all the chunks when only
    # small batches sizes are wanted for testing.
	j = 0
	for chunk in essays_labels:
	    data_app = data_app.append(chunk[chunk.got_posted=='t'])
	    data_rej = data_rej.append(chunk[chunk.got_posted!='t'])
	    j=j+1
	    if breakme and j >= MaxChunks:
		break
	return data_app,data_rej,headers

data_app,data_rej,headers = RunItTimeIt(LoadByChunking,
	[file_essays_labels2,True],True,
	"Load data by chunking,")
################### END CHUNKING


# <<<<< MAY NEED TO BE REVISED IN FUTURE >>>>>
# for now, just get rid of rows with missing labels.  there aren't that
# many anyway.
data_app = data_app[data_app.got_posted.isnull()==False]
data_rej = data_rej[data_rej.got_posted.isnull()==False]

# reset indices in both data frames
#data_app = data_app.reset_index()
#data_rej = data_rej.reset_index()

# change labels to 1 and 0
data_app = data_app.replace(to_replace={'got_posted':{'t':0,'f':1}})
data_rej = data_rej.replace(to_replace={'got_posted':{'t':0,'f':1}})

# TRAIN/TEST split
data_app_train, data_app_test = train_test_split(data_app,test_size=mytest_size)
data_rej_train, data_rej_test = train_test_split(data_rej,test_size=mytest_size)

data_app_train = pd.DataFrame(data_app_train,columns=headers)
data_app_test = pd.DataFrame(data_app_test,columns=headers)
data_rej_train = pd.DataFrame(data_rej_train,columns=headers)
data_rej_test = pd.DataFrame(data_rej_test,columns=headers)


############################# Generate features. 
#   See RunItTimeIt description above
train_app = RunItTimeIt(generate_features,[data_app_train],True,
       "finished generating features: training approved,")
#train_app = generate_features(data_app_train)

test_app = RunItTimeIt(generate_features,[data_app_test],True,
       "finished generating features: test approved,")
train_rej = RunItTimeIt(generate_features,[data_rej_train],True,
       "finished generating features: training rejected,")
test_rej = RunItTimeIt(generate_features,[data_rej_test],True,
       "finished generating features: test rejected,")


############################# Combine training data and train classifier
train = train_app + train_rej
test = test_app + test_rej

classifier = RunItTimeIt(NaiveBayesClassifier.train,[train],True,
       "Train Naive Bayes Classifier,")

############################# Test accuracies
print ('Test approved accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_app)))
print ('Test rejected accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_rej)))


probs = []
actual = []
for item in test:
    probs.append(classifier.prob_classify(item[0]).prob(1))
    actual.append(item[1])
probs = np.array(probs)
actual = np.array(actual)
fpr,tpr,_ = roc_curve(y_true=actual,y_score=probs)
roc_auc = auc(lr_fpr,lr_tpr)




'''
logreg.fit(train[train.columns[train.columns!='Y']],train.Y)
lr_fitresults = logreg.predict(test[test.columns[test.columns!='Y']])
lr_confidence = logreg.decision_function(test[test.columns[test.columns!='Y']])
lr_fpr,lr_tpr,_ = roc_curve(y_true=Y,y_score=lr_confidence)
lr_roc_auc = auc(lr_fpr,lr_tpr)
end = time.time()
'''

print "Plot Results"
plt.figure()
plt.plot(fpr,tpr,label="NaiveBayes (AUC = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()


'''
totalpos = sum(actual)
totalneg = len(actual)-totalpos
fnr = totalpos*(1-tpr)/totalneg
tnr = 1- fpr
print "Plot Results"
plt.figure()
plt.plot(fnr,tnr,label="NaiveBayes (AUC = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Negative Rate')
plt.ylabel('True Negative Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
'''
