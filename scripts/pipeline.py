'''
pipeline.py
'''

from cleanResultantMerge import *

'''
clean up and downsample resultant merge, and save summary statistics
'''

filen = "../data/resultant_merge.csv"
rawdf = pd.read_csv(filen)
df = cleanData(rawdf)
dsdf = downSample(df, 3)
dsdfSummary = getSummary(dsdf)
dsdfSummary.to_csv('../data/summary_stats.csv', index=False)
dsdf.to_csv('../data/clean_labeled_project_data.csv', index=False)


'''
merge all_essays.csv with cleaned up project data from previous step
'''
from DataMerge import *
MergeLabelsAndEssays()