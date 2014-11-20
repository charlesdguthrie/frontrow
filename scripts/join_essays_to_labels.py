#join_essays_to_labels
from sys import argv
import pandas as pd
import csv
from DataLoading import *

script, second = argv

labeled_inpath = '../data/essays_and_labels.csv'
full_essays_path = '../data/opendata_essays.csv'
outpath = '../data/' + second

print "loading " + labeled_inpath
labeled_df=pd.read_csv(labeled_inpath,sep=',', header=0)
print "finished loading " + labeled_inpath


out_columns = labeled_df.columns

#initialize csv file
with open(outpath, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(out_columns)

def prep_data(chunk,chunk_id):
    chunk.columns = ['_projectid','_teacher_acctid','title','short_description','need_statement','essay','paragraph1','paragraph2','paragraph3','paragraph4','thankyou_note','impact_letter']
    print "chunk",chunk_id
    m,n = chunk.shape

    outchunk = pd.DataFrame(columns=out_columns)
    for i in range(m):
        row_id = chunk_id*m+i
        row = chunk.irow(i)
        finished_line = join_essays_to_labels(row,row_id)
        outchunk = outchunk.append(finished_line, ignore_index=True)

    write_chunk(outpath,outchunk)
 
    
#write processed output one line at a time.  
def write_line(outpath,outrow):
    with open(outpath, 'a') as f:
        outrow.to_csv(f, header=False, index=False)   
     
def write_chunk(outpath,outchunk):
    with open(outpath, 'a') as f:
        outchunk.to_csv(f, header=False, index=False)  


def join_essays_to_labels(row, row_id):

    pid = row[0].replace('"', '').strip()

    essay = row['essay']

    labeled_df['essay'][labeled_df['_projectid']==pid] = essay
    myline = labeled_df[labeled_df['_projectid']==pid]
    return myline

chunker(full_essays_path,prep_data,10000)
