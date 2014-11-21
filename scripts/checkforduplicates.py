import pandas as pd
from DataLoading import *
#from sqlalchemy import create_engine

           

'''
filename = "labelled_projectids_jmj1.csv"
filepath = getDataFilePath(filename)
labelled_projectids = pd.read_csv(filepath)

ids = labelled_projectids._projectid.unique()

countids = pd.DataFrame(zip(ids,np.zeros(len(ids))),columns=["projectid","count"])

print labelled_projectids.got_posted.value_counts()
print "# duplicate projectids:",sum(labelled_projectids._projectid.value_counts()>1)

print "# duplicate essays:",sum(labelled_projectids._projectid.value_counts()>1)
'''


def getNumDuplicates(series):
    valcounts = series.value_counts()
    dups = valcounts[valcounts>1]
    return sum(dups)
    
    
    
filename = "essays_and_labels.csv"
filepath_Labels = getDataFilePath(filename)
df = pd.read_csv(filepath_Labels)     

print "# duplicate projectids:",getNumDuplicates(df._projectid)
print "# duplicate titles:",getNumDuplicates(df.title)
print "# duplicate need statements:",getNumDuplicates(df.need_statement)
print "# duplicate essays:",getNumDuplicates(df.essay)



valcounts = df.essay.value_counts()
dup_vals = valcounts[valcounts>1]
dup_ind = df.essay.apply(lambda x: x in dup_vals)
dups = df[dup_ind]

app = sum(dups.got_posted=='t')
rej = sum(dups.got_posted=='f')
print "Pivoting on duplicate essays:"
print "# Approved:",app
print "# Rejected:",rej
print "# Other:",len(dups)-app-rej
