import pandas as pd
import sklearn as sk
#import sklearn.linear_model as lm
import nltk
from nltk import NaiveBayesClassifier
import nltk.classify
from sklearn.cross_validation import train_test_split
import os
from utils import *
from FeatureGeneration import *
from Statistics import *
from DataLoading import *

mydir = os.path.dirname(os.path.realpath(__file__))
pardir = os.path.join(mydir,"..")
datadir = os.path.join(pardir,"data")

file_essays = os.path.join(datadir,"full_labeled_essays_1000.csv")
file_essays_labels = os.path.join(datadir,"full_labeled_essays.csv")
file_essays_labels2 = os.path.join(datadir,"essays_and_labels.csv")
file_resources = os.path.join(datadir,"opendata_resources.csv")


data_app,data_rej,headers = LoadByChunking(file_essays_labels2,breakme=True,MaxChunks=1)


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
mytest_size = 0.3
data_app_train, data_app_test = train_test_split(data_app,test_size=mytest_size)
data_rej_train, data_rej_test = train_test_split(data_rej,test_size=mytest_size)

data_app_train = pd.DataFrame(data_app_train,columns=headers)
data_app_test = pd.DataFrame(data_app_test,columns=headers)
data_rej_train = pd.DataFrame(data_rej_train,columns=headers)
data_rej_test = pd.DataFrame(data_rej_test,columns=headers)


train_app = NLTKfeatures(data_app_train,False,msg="train_app")
test_app = NLTKfeatures(data_app_test,False,msg="test_app")
train_rej = NLTKfeatures(data_rej_train,False,msg="train_rej")
test_rej = NLTKfeatures(data_rej_test,False,msg="test_rej")


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


fpr,tpr,roc_auc = getROC_NLTK(classifier,test)
    
print "Plot Results"
plotROC(fpr,tpr,roc_auc)


'''
totalpos = sum(actual)
totalneg = len(actual)-totalpos
fnr = totalpos*(1-tpr)/totalneg
tnr = 1- fpr

'''
