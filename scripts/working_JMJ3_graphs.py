import pandas as pd
import numpy as np
import DataSets as ds
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import DataLoading as dl

#tableau color palette
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.) 


def makehist(series,rejected,mincount=0,bins=[],autobin=False,title="",filename="_"):
    rej = rejected == 1
    app = rejected == 0
    nrej = sum(rej)*1.0
    napp = sum(app)*1.0
    
    
    #series_rej = pd.Series({count: sum(series[rej]==count) for count in pd.unique(series[rej])})
    #series_app = pd.Series({count: sum(series[app]==count) for count in pd.unique(series[app])})

    #rej_plot = series_rej[series_rej.index>=mincount]/nrej
    #app_plot = series_app[series_app.index>=mincount]/napp
    fig, ax = plt.subplots(1)
    
    if autobin == False:
        if len(bins)==0:
            bins = np.arange(mincount,mincount+40,5)
        ppl.hist(ax,np.array(series[rej]),bins=bins,label='rejected',color=tableau20[2],alpha=0.8)
        ppl.hist(ax,np.array(series[app]),bins=bins,label='approved',color=tableau20[15],alpha=0.4)
        
            
        #n1,bin1,_ = ppl.hist(ax,np.array(rej_plot.index),bins=bins,weights=np.array(rej_plot),label='rejected')
        #n2,bin2,_ = ppl.hist(ax,np.array(app_plot.index),bins=bins,weights=np.array(app_plot),label='approved')
    else:        
        ppl.hist(ax,np.array(series[rej]),label='rejected',color=tableau20[2],alpha=0.8)
        ppl.hist(ax,np.array(series[app]),label='approved',color=tableau20[15],alpha=0.4)
        
        #n1,bin1,_ = ppl.hist(ax,np.array(rej_plot.index),weights=np.array(rej_plot),label='rejected')
        #n2,bin2,_ = ppl.hist(ax,np.array(app_plot.index),weights=np.array(app_plot),label='approved')
    plt.legend()
    if mincount > 0:
        title = title + "(count >= " + str(mincount) + ")"
    plt.title(title)
    
    filepath = dl.getDataFilePath('plots/fig_'+filename+'.png')
    plt.savefig(filepath)
    
    plt.show()
    
    #df_freq = pd.concat([
    #                pd.DataFrame(series_rej/nrej,columns=['rejected']),
    #                pd.DataFrame(series_app/napp,columns=['approved'])],axis=1)
    #return df_freq
    
def makebarplot(series,rejected,title="",filename='_'):
    rej = rejected == 1
    app = rejected == 0
    nrej = sum(rej)*1.0
    napp = sum(app)*1.0
    
    ser_rej = series[rej]
    ser_app = series[app]
    
    ser_rej_T = sum(ser_rej==True)
    #ser_rej_F = sum(ser_rej==False)
    ser_app_T = sum(ser_app==True)
    #ser_app_F = sum(ser_app==False)
    
    ser_rej_plot = [ser_rej_T/nrej]
    ser_app_plot = [ser_app_T/napp]
    #ser_plot = [ser_rej_T,ser_app_T]
    
    #width = 0.35
    ind = np.arange(2)
    fig, ax = plt.subplots()
    ppl.bar(ax,ind,
            np.array([ser_rej_plot,ser_app_plot]),
            xticklabels=('rejected('+str(ser_rej_T)+')','approved('+str(ser_app_T)+')'),
            annotate=False)
    #ax.set_xticks(ind+width)
    #ax.set_xticks((width/2.0,width*3/2.0))    
    #ax.set_xticklabels(('rejected('+str(ser_rej_T)+')','approved('+str(ser_app_T)+')'))
    plt.title(title)
    plt.ylabel('% of subset rejected/approved')
    #plt.legend(loc="lower right")
    filepath = dl.getDataFilePath('plots/fig_'+filename+'.png')
    plt.savefig(filepath)
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
essay_len = df.essay_len
total_price = df.total_price_excluding_optional_support
pay_proc = df.payment_processing_charges


#bins = [4,8,12,16,20,24]
makehist(
    maxcaps,
    rejected,
    mincount=4,
    title="Max consecutive capitilized letters",
    filename='maxcaps')
                
#bins = [1,6,11,16,21,26,31,36]
makehist(
    totalcaps,
    rejected,
    mincount=1,
    title="total # capitilized letters",
    filename='totalcaps')
                
makehist(
    dollarcount_ser,
    rejected,
    mincount=1,
    title="total # '$'",
    filename='dollarcount')
                
makebarplot(
    dollarbool_ser,
    rejected,
    title="$ present",
    filename='dollarbool')
    
makebarplot(
    email_ser,
    rejected,
    title="@ present",
    filename='emailbool')
    
makebarplot(
    urls_ser,
    rejected,
    title="url present",
    filename='urlbool')
    
bins = np.arange(0,3501,50)
makehist(
    essay_len,
    rejected,
    mincount=0,
    bins=bins,
    title="Essay Length",
    filename='essay_len')
                
bins = np.arange(0,3501,50)
makehist(
    total_price,
    rejected,
    mincount=0,
    bins=bins,
    title="Total Price",
    filename='total_price')
                
bins = np.arange(0,41,2)
makehist(
    pay_proc,
    rejected,
    mincount=0,
    bins=bins,
    title="Payment Processing Charges",
    filename='pay_proc')