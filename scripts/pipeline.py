'''
pipeline.py

to run this you need resultant_merge.csv and all_essays.csv in your data directory
'''

from cleanResultantMerge import *
from DataMerge import MergeLabelsAndEssays
from DataSets import *


#clean up and downsample resultant merge, and save summary statistics

filen = "../data/resultant_merge.csv"
rawdf = pd.read_csv(filen)
df = cleanData(rawdf)
dsdf = downSample(df, 3)
dsdfSummary = getSummary(dsdf)
dsdfSummary.to_csv('../data/summary_stats.csv', index=False)
dsdf.to_csv('../data/clean_labeled_project_data.csv', index=False)



#merge all_essays.csv with cleaned up project data from previous step
MergeLabelsAndEssays()

#pickle and vectorize the data
ImportPickleBalancedFull()
PickleVectorized()


#Results will be in:
'''
'BalancedFull'
'BalancedFull_Essay_Vectorized'
'BalancedFull_NeedStatement_Vectorized'
'''