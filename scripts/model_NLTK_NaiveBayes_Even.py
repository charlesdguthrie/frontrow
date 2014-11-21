import pandas as pd
import nltk.classify

from utils import *
from FeatureGeneration import *
from Statistics import *
from DataLoading import *

from sklearn.cross_validation import train_test_split
import gc

gc.collect()
filename = "essays_and_labels.csv"
data_app_raw,data_rej = getChunkedData(filename,breakme=False)

n_app = len(data_app_raw)
n_rej = len(data_rej)
ratio = n_rej*1.0/n_app
data_app_train, data_app_ignore = train_test_split(data_app_raw,test_size=1-ratio)
data_app = pd.DataFrame(data_app_train,columns=data_app_raw.columns)

print "**********************************"
print "BALANCED DATA:"
print "approved data set:"
print "   approved =",np.sum(data_app.got_posted==1)
print "   rejected =",np.sum(data_app.got_posted==0)
print "rejected data set:"
print "   approved =",np.sum(data_rej.got_posted==1)
print "   rejected =",np.sum(data_rej.got_posted==0)
print "**********************************"


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

fpr,tpr,roc_auc,thresholds = getROC_NLTK(classifier,test)
    
print "Plot Results"
plotROC(fpr,tpr,roc_auc)

#### Use this to check that labels are converted correctly
filepath = getDataFilePath(filename)
dfa,dfr,headers = LoadByChunking(filepath,breakme=True,MaxChunks=5)

print "Approved:",dfa.got_posted[0],"->",data_app.got_posted[0]
print "Rejected:",dfr.got_posted[0],"->",data_rej.got_posted[0]


plt.figure()
plt.plot(thresholds,fpr,label="FPR")
plt.plot(thresholds,tpr,label="TPR")
plt.legend()
plt.xlabel("thresholds")
plt.title("FPR & TPR")
plt.show()



