"""
Created on Sun Nov 23

@author: charlesdguthrie
"""

import pandas as pd
import numpy as np


'''
replace blank strings with nulls
replace 1970 dates with nulls
remove null 'got_posted'
create 'rejected' field, where 1 = rejected, 0 = approved
create variables to indicate nulls
eliminate too-recent data after 10/31
'''
def cleanData(rawdf):
    df=rawdf
    
    #remove quotes from project id, teacher id, school id
    df._projectid = df._projectid.str.replace('"','')
    df._teacher_acctid = df._teacher_acctid.str.replace('"','')
    df._schoolid = df._schoolid.str.replace('"','')

    #Replace blank strings with nulls
    df.replace("",np.nan, inplace=True)
    
    #Get rid of values after 10/31 and before 2008
    df = df[(df.date_posted.isnull()) | ((df.date_posted>='2008-01-01') & (df.date_posted < '2014-10-31'))]
    
    #remove rows where got_posted is null
    df = df[pd.notnull(df.got_posted)]
    
    #remove leakage columns
    df.drop(['students_reached','total_donations','num_donors','eligible_double_your_impact_match','eligible_almost_home_match',
             'funding_status','date_completed','date_thank_you_packet_mailed','date_expiration','date_posted'],inplace=True,axis=1)
    
    
    #create 'rejected' field, where 1 = rejected, 0 = approved
    df['rejected'] = np.where(df.got_posted == 'f', 1,0)
    df = df.drop('got_posted', 1)

    #Create null value column
    for col in df.columns:
        # if there are any nulls
        if len(df[col][pd.isnull(df[col])])>0:
            df[col+'_mv'] = np.where(pd.isnull(df[col]),1,0)
        #f->0, t->1
        if df[col].dtype=='O':
            df = df.replace(to_replace={col:{'f':0,'t':1}})


    #Drop duplicates
    df = df.drop_duplicates()
    return df

'''
bring ratio of approvals:rejections down to desired level
'''
def downSample(df, app_rej_ratio):
    app=df[df.rejected==0]
    rej=df[df.rejected==1]
    
    #get it down to 10 approvals for every rejection
    rand = np.random.choice(app.shape[0], size=app_rej_ratio*rej.shape[0], replace=False)
    DownSample = app.iloc[rand]
    
    outdf = rej.append(DownSample, ignore_index=True)
    #outdf.drop_duplicates(inplace=True)
    return outdf

def filterDates(df):
    return df[(df.created_date>='2008-01-01') & (df.created_date<'2014-10-31') & (pd.notnull(df.created_date))]

def splitOnDateAndDownSample(df,myDate):
    df2 = df
    df2['train']=np.where(df2.created_date<myDate,1,0)
    train = df2[df2.train==1]
    test = df2[df2.train==0]
    trainds = downSample(train,1)
    testds = downSample(test,1)

    return pd.concat([trainds,testds],axis=0, ignore_index=True)


'''
get number of nulls, number of unique values, and ten most common values
'''
def getSummary(df,rejected):
    uniqueList = []
    typeList = []
    valueList = []
    iList = []
    meanList = []
    mean_rejList = []
    mean_appList = []
    stdList = []
    
    for i,col in enumerate(df.columns):
        uniques = len(df[col][df[col].notnull()].unique())
        uniqueList.append(uniques)
        if(uniques>100):
            values = "too many to calculate"
        else:
            numvals = len(df[col].value_counts())
            values = df[col].value_counts().iloc[:min(10,numvals)]
        valueList.append(values)
        typeList.append(np.dtype(df[col]))
        iList.append(i)
        meanList.append(df[col].mean())
        mean_rejList.append(df[rejected==1][col].mean())
        mean_appList.append(df[rejected==0][col].mean())
        stdList.append(df[col].std())
        
    uniqueSeries = pd.Series(uniqueList, index=df.columns)
    meanSeries = pd.Series(meanList,index=df.columns)
    meanrejSeries = pd.Series(mean_rejList,index=df.columns)
    meanappSeries = pd.Series(mean_appList,index=df.columns)
    stdSeries = pd.Series(stdList,index=df.columns)
    valueSeries = pd.Series(valueList, index=df.columns)
    typeSeries = pd.Series(typeList, index=df.columns)
    iSeries = pd.Series(iList,index=df.columns)
    
    #uniques
    summaryItems = [
        ('nulls', df.shape[0] - df.count()),
        ('distinct_count', uniqueSeries),
        ('mean', meanSeries),
        ('mean_rej', meanrejSeries),
        ('mean_app', meanappSeries),
        ('std_dev', stdSeries),
        ('top10Values', valueSeries),
        ('dtype', typeSeries),
        ('i', iSeries)
    ]
    summaryDF = pd.DataFrame.from_items(summaryItems)
    #print 'Rows,Columns',df.shape
    return summaryDF


# filen = "../data/resultant_merge.csv"
# rawdf = pd.read_csv(filen)
# df = cleanData(rawdf)
# dsdf = downSample(df, 3)
# dsdfSummary = getSummary(dsdf)
# dsdfSummary.to_csv('../data/summary_stats.csv', index=False)
# dsdf.to_csv('../data/clean_labeled_project_data.csv', index=False)