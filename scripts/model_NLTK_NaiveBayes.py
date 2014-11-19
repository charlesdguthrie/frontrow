import pandas as pd
import nltk.classify

from utils import *
from FeatureGeneration import *
from Statistics import *
from DataLoading import *


file_essays_labels2 = "essays_and_labels.csv"

data_app,data_rej = getChunkedData(file_essays_labels2,breakme=True,MaxChunks=1)
data_app_train,data_app_test,data_rej_train,data_rej_test = TrainTestSplit(data_app,data_rej)

train_app = NLTKfeatures(data_app_train,lemmatize=False,msg="train_app")
test_app = NLTKfeatures(data_app_test,lemmatize=False,msg="test_app")
train_rej = NLTKfeatures(data_rej_train,lemmatize=False,msg="train_rej")
test_rej = NLTKfeatures(data_rej_test,lemmatize=False,msg="test_rej")

train = train_app + train_rej
test = test_app + test_rej

classifier = Classifier_NLTK_NaiveBayes(train)

############################# Test accuracies
print ('Test approved accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_app)))
print ('Test rejected accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_rej)))

fpr,tpr,roc_auc = getROC_NLTK(classifier,test)
    
print "Plot Results"
plotROC(fpr,tpr,roc_auc)

