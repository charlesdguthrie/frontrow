'''
pipeline.py

to run this you need resultant_merge.csv and all_essays.csv in your data directory
run this script from your script directory
'''

from cleanResultantMerge import *
from DataMerge import MergeToFull
from DataSets import *


#clean up and downsample resultant merge, and save summary statistics

filen = "../data/resultant_merge.csv"

print "reading resultant_merge.csv..."
rawdf = pd.read_csv(filen)
print "read complete"



print "Cleaning resultant merge..."
df = cleanData(rawdf)
#dfSummary = getSummary(df)
#dfSummary.to_csv('../data/summary_stats.csv', index=True)
df.to_csv('../data/clean_labeled_project_data.csv', index=False)
print "Cleaning complete"

print "Merging created_date to data..."
extractFileName = "essays_and_labels.csv"
outFileName = "data_with_dates.csv"
extractedCols = ['_projectid', 'created_date']
MergeToFull(extractFileName,df,outFileName,extractedCols)

print "Reading... "
outpath = "../data/" + outFileName
df = pd.read_csv(outpath)

print "Filtering dates..."
df = filterDates(df)

print "Splitting on created_date and downsampling..."
df = splitOnDateAndDownSample(df,'2013-05-01')

#merge all_essays.csv with cleaned up project data from previous step
print "Merging essays to data..."
extractFileName = "all_essays.csv"
outFileName = "data_with_dates.csv"
extractedCols = ['_projectid', 'title', 'short_description', 'need_statement', 'essay']
MergeToFull(extractFileName,df,outFileName,extractedCols)
print "Merged essays"

print "Reading again..."
df = pd.read_csv("../data/" + outFileName)

print "Merges complete."

#pickle and vectorize the data
print "Pickling merged data..."
ImportPickleBalancedFull(df)
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