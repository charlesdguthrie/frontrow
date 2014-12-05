"""
Created on Sun Nov 23

@author: charlesdguthrie
"""

import pandas as pd
import numpy as np

filen = "../data/resultant_merge.csv"
rawdf = pd.read_csv(filen)

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

'''
get number of nulls, number of unique values, and ten most common values
'''
def getSummary(df):
    uniqueList = []
    typeList = []
    valueList = []
    
    for col in df.columns:
        uniques = len(df[col][df[col].notnull()].unique())
        uniqueList.append(uniques)
        if(uniques>100):
            values = "too many to calculate"
        else:
            values = df[col].value_counts().iloc[:10]
        valueList.append(values)
        typeList.append(np.dtype(df[col]))
        
    uniqueSeries = pd.Series(uniqueList, index=df.columns)
    valueSeries = pd.Series(valueList, index=df.columns)
    typeSeries = pd.Series(typeList, index=df.columns)
    
    #uniques
    summaryItems = [
        ('nulls', df.shape[0] - df.count()),
        ('distinct_count', uniqueSeries),
        ('top10Values', valueSeries),
        ('dtype', typeSeries)
    ]
    summaryDF = pd.DataFrame.from_items(summaryItems)
    print 'Rows,Columns',df.shape
    return summaryDF



df = cleanData(rawdf)
dsdf = downSample(df, 3)
dsdfSummary = getSummary(dsdf)
dsdfSummary.to_csv('../data/summary_stats.csv', index=False)
dsdf.to_csv('../data/clean_labeled_project_data.csv', index=False)