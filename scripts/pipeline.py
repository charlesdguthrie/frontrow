'''
pipeline.py

to run this you need resultant_merge.csv and all_essays.csv in your data directory
'''

from cleanResultantMerge import *
from DataMerge import MergeLabelsAndEssays
from DataSets import *


#clean up and downsample resultant merge, and save summary statistics

filen = "../data/resultant_merge.csv"

print "reading resultant_merge.csv..."
rawdf = pd.read_csv(filen)
print "read complete"



print "Cleaning resultant merge..."
df = cleanData(rawdf)
dsdf = downSample(df, 1) #second argument is the approval-to-rejection ratio.  1 produces a 50-50 split
dsdfSummary = getSummary(dsdf)
dsdfSummary.to_csv('../data/summary_stats.csv', index=True)
dsdf.to_csv('../data/clean_labeled_project_data.csv', index=False)
print "Cleaning complete"


#merge all_essays.csv with cleaned up project data from previous step
print "Merging essays with data..."
MergeLabelsAndEssays()
print "Merge complete."

#pickle and vectorize the data
print "Pickling merged data..."
ImportPickleBalancedFull()
print "Pickle complete."

print "Vectorizing essays and need statements..."
PickleVectorized()
print "Vectorizing complete"

print """
Results will be in:
data/BalancedFull.pk1
data/BalancedFull_Essay_Vectorized.pk1
data/BalancedFull_NeedStatement_Vectorized.pk1
"""