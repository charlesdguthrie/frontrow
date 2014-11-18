#join_essays_to_labels
from sys import argv
import pandas as pd
import csv

script, second = argv

labeled_inpath = '../data/essays_and_labels.csv'
full_essays_path = '../data/opendata_essays_2500-3000.csv'
my_outpath = '../data/' + second

print "loading " + labeled_inpath
labeled_df=pd.read_csv(labeled_inpath,sep=',', header=0)
print "finished loading " + labeled_inpath

chunksize = 500
out_columns = ['got_posted','_projectid','_teacherid','title','essay','need_statement','created_date']

def prep_data(inpath,outpath):

    #initialize csv file
    with open(outpath, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(out_columns)

    #read lines one at a time
    with open(inpath, 'rb') as inf:
        reader = pd.read_csv(inf, delimiter=',', quotechar='"',iterator=True, chunksize=chunksize)

        #Read and print header
        #header = inf.readline().split(',')

        #Read and process other lines
        for chunk_id,chunk in enumerate(reader):
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
    if myline.shape[1]>7:
        print "extra wide row:",row_id
    return myline

prep_data(full_essays_path,my_outpath)

