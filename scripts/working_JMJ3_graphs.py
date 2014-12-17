import pandas as pd
import numpy as np
import DataSets as ds
import matplotlib.pyplot as plt
import prettyplotlib as ppl


def makehist(series,rejected,mincount=0,bins=[],title=""):
    rej = rejected == 1
    app = rejected == 0
    nrej = sum(rej)*1.0
    napp = sum(app)*1.0
    
    series_rej = pd.Series({count: sum(series[rej]==count) for count in pd.unique(series[rej])})
    series_app = pd.Series({count: sum(series[app]==count) for count in pd.unique(series[app])})

    rej_plot = series_rej[series_rej.index>=mincount]/nrej
    app_plot = series_app[series_app.index>=mincount]/napp
    fig, ax = plt.subplots(1)
    if len(bins)>0:
        n1,bin1,_ = ppl.hist(ax,np.array(rej_plot.index),bins=bins,weights=np.array(rej_plot),label='rejected')
        n2,bin2,_ = ppl.hist(ax,np.array(app_plot.index),bins=bins,weights=np.array(app_plot),label='approved')
    else:
        n1,bin1,_ = ppl.hist(ax,np.array(rej_plot.index),weights=np.array(rej_plot),label='rejected')
        n2,bin2,_ = ppl.hist(ax,np.array(app_plot.index),weights=np.array(app_plot),label='approved')
    plt.legend()
    if mincount > 0:
        title = title + "(count >= " + str(mincount) + ")"
    plt.title(title)
    plt.show()
    
    df_freq = pd.concat([
                    pd.DataFrame(series_rej/nrej,columns=['rejected']),
                    pd.DataFrame(series_app/napp,columns=['approved'])],axis=1)
    return df_freq
    
def makebarplot(series,rejected,title=""):
    rej = rejected == 1
    app = rejected == 0
    nrej = sum(rej)*1.0
    napp = sum(app)*1.0
    
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



# LOAD DATA
dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad('FeatureSet_A')
df = dense_df
maxcaps = df.maxcaps
totalcaps = df.totalcaps
dollarcount_ser = df.dollarcount
dollarbool_ser = df.dollarbool
email_ser = df.email
urls_ser = df.urls


maxcaps_df = makehist(maxcaps,rejected,mincount=4,title="Max consecutive capitilized letters")
bins = [0,4,8,20,500]
totalcaps_df = makehist(totalcaps,rejected,mincount=1,title="total # capitilized letters")
dollarcount_df = makehist(dollarcount_ser,rejected,mincount=1,title="total # '$'")
makebarplot(dollarbool_ser,rejected,title="$ present")
makebarplot(email_ser,rejected,title="@ present")
makebarplot(urls_ser,rejected,title="url present")
