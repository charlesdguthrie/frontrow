import pandas as pd
import numpy as np

from Statistics import *
import TextProcessing as txtpr
from DataSets import *
from FeatureGeneration import *
from DataLoading import *

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc


filename = "Merge_2014_12_05.csv"
filepath = getDataFilePath(filename)
df = pd.read_csv(filepath)
'''
chunksize = 50000
filename = "all_essays.csv"
filepath = getDataFilePath(filename)
chunker = pd.read_csv(filepath,iterator=True,chunksize=chunksize)
chunk = chunker.get_chunk(chunksize)

'''


def ShoutingCount(df_column):
    def IdentifyShouting(words):
        if words is nan:
            return 0,0
        else:
            words = txtpr.RemoveSymbolsAndSpecial(words)
            words = words.split()
            allcaps = [x.isupper() for x in words]
            totalcaps = sum(allcaps)
            maxconsecutivecaps = 0
            count = 0
            for x in allcaps:
                if x:
                    count += 1
                    maxconsecutivecaps = max(count,maxconsecutivecaps)
                else:
                    count = 0
            return totalcaps,maxconsecutivecaps
    shouting = [IdentifyShouting(words) for words in df_column.fillna('')]
    return np.array(shouting)


def containsDollarSign(df_column,boolean=True):
    if boolean:
        return np.array(['$' in words for words in df_column.fillna('')])
    else:
        return np.array([words.count('$') for words in df_column.fillna('')])
        
def containsEmailAddress(df_column):
    return np.array(['@' in words for words in df_column.fillna('')])

def containsURL(df_column):
    def findURL1(words):
        return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', words))>0
    def findURL2(words):
        return 'www.' in words or '.com' in words or '.org' in words or 'htm' in words or '.edu' in words 
    return np.array([findURL1(words) or findURL2(words) for words in df_column.fillna('')])

essays = df.essay.copy()
shouting = pd.DataFrame(ShoutingCount(essays),columns=['totalcaps','max_consecutive_caps'])
dollarbool = containsDollarSign(essays)
dollarcount = containsDollarSign(essays,boolean=False)
email = containsEmailAddress(essays)
urls = containsURL(essays)

maxcaps = pd.Series(shouting[:,1])
totalcaps = pd.Series(shouting[:,0])
dollarbool_ser = pd.Series(dollarbool)
dollarcount_ser = pd.Series(dollarcount)
email_ser = pd.Series(email)
urls_ser = pd.Series(urls)
                
def makehist(series,df,mincount=0,bins=[],title=""):
    rej = df.rejected == 1
    app = df.rejected == 0
    nrej = sum(rej)*1.0
    napp = sum(app)*1.0
    
    series_rej = pd.Series({count: sum(series[rej]==count) for count in pd.unique(series[rej])})
    series_app = pd.Series({count: sum(series[app]==count) for count in pd.unique(series[app])})

    rej_plot = series_rej[series_rej.index>=mincount]/nrej
    app_plot = series_app[series_app.index>=mincount]/napp
    plt.figure()
    if len(bins)>0:
        n1,bin1,_ = plt.hist(np.array(rej_plot.index),bins=bins,weights=np.array(rej_plot),label='rejected')
        n2,bin2,_ = plt.hist(np.array(app_plot.index),bins=bins,weights=np.array(app_plot),label='approved')
    else:
        n1,bin1,_ = plt.hist(np.array(rej_plot.index),weights=np.array(rej_plot),label='rejected')
        n2,bin2,_ = plt.hist(np.array(app_plot.index),weights=np.array(app_plot),label='approved')
    plt.legend()
    if mincount > 0:
        title = title + "(count >= " + str(mincount) + ")"
    plt.title(title)
    plt.show()
    
    df_freq = pd.concat([
                    pd.DataFrame(series_rej/nrej,columns=['rejected']),
                    pd.DataFrame(series_app/napp,columns=['approved'])],axis=1)
    return df_freq
    
def makebarplot(series,df,title=""):
    ser_rej = series[rej]
    ser_app = series[app]
    
    ser_rej_T = sum(ser_rej==True)
    ser_rej_F = sum(ser_rej==False)
    ser_app_T = sum(ser_app==True)
    ser_app_F = sum(ser_app==False)
    
    ser_rej_plot = [ser_rej_T/nrej]
    ser_app_plot = [ser_app_T/napp]
    ser_plot = [ser_rej_T,ser_app_T]
    
    width = 0.35
    ind = np.arange(1)
    fig, ax = plt.subplots()
    ax.bar(ind,ser_rej_plot,width,color='r',label="rejected")
    ax.bar(ind+width,ser_app_plot,width,color='b',label="approved")
    #ax.set_xticks(ind+width)
    ax.set_xticks((width/2.0,width*3/2.0))    
    ax.set_xticklabels(('rejected('+str(ser_rej_T)+')','approved('+str(ser_app_T)+')'))
    plt.title(title)
    #plt.legend(loc="lower right")
    plt.show()

maxcaps_df = makehist(maxcaps,df,mincount=4,title="Max consecutive capitilized letters")
bins = [0,4,8,20,500]
totalcaps_df = makehist(totalcaps,df,mincount=1,title="total # capitilized letters")
dollarcount_df = makehist(dollarcount_ser,df,mincount=1,title="total # '$'")
makebarplot(dollarbool_ser,df,title="$ present")
makebarplot(email_ser,df,title="@ present")
makebarplot(urls_ser,df,title="url present")