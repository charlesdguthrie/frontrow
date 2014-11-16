#join_essays_to_labels
import pandas as pd
import csv


labeled_inpath = '../data/essays_and_labels.csv'
full_essays_path = '../data/opendata_essays.csv'
my_outpath = '../data/full_labeled_essays.csv'

print "loading " + labeled_inpath
labeled_df=pd.read_csv(labeled_inpath,sep=',', header=0)
print "finished loading " + labeled_inpath

def prep_data(inpath,outpath):

    #initialize csv file
    with open(outpath, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['got_posted','_projectid','_teacherid','title','essay','need_statement','created_date'])

    #read lines one at a time
    with open(inpath, 'rb') as inf:
        reader = pd.read_csv(inf, delimiter=',', quotechar='"',iterator=True, chunksize=100)
        
        #Read and print header
        #header = inf.readline().split(',')

        #Read and process other lines
        for chunk in reader:
            m,n = chunk.shape
            for i in range(m):
                row = chunk.irow(i)
                prep_line(row)
                           
        #for row in csvreader:
        #    prep_line(header,row)
            

#process one line.  Produces a dict with key:header, value: data   
def prep_line(row):
    row[0] = row[0].replace('"', '').strip()
    finished_line = join_essays_to_labels(row['_projectid'],row['essay'])
    write_line(my_outpath, finished_line.iloc[:,:6])
    '''
    -----------------------------------
    Insert text processing modules here, eg:
    punct_free = remove_punctuation(line_dict['essay'])
    stopwords_free = remove_stopwords(punct_free)
    stemmed = stem(stopwords_free)
    -----------------------------------
    '''
    
#write processed output one line at a time.  
def write_line(outpath,outrow):
    with open(outpath, 'a') as f:
        outrow.to_csv(f, header=False, index=False)   
        
def join_essays_to_labels(pid, essay):

    labeled_df['essay'][labeled_df['_projectid']==pid] = essay
    myline = labeled_df[labeled_df['_projectid']==pid
    if len(myline)>7:
        print myline
    return myline

prep_data(full_essays_path,my_outpath)

