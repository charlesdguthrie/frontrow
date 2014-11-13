#join_essays_to_labels
import pandas as pd

homedir = '/Users/205341/Documents/git/frontrow/'
labeled_inpath = homedir + 'essays_and_labels.csv'
full_essays_path = homedir + 'opendata_essays_1000.csv'
my_outpath = homedir + 'full_labeled_essays.csv'

print "loading " + labeled_inpath
labeled_df=pd.read_csv(labeled_inpath,sep=',', header=0)
print "finished loading " + labeled_inpath
labeled_df["full_essay"]=""


def prep_data(inpath,outpath):
    import csv

    #initialize csv file
    with open(outpath, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['got_posted','_projectid','_teacherid','title','essay','need_statement','created_date','full_essay'])

    #read lines one at a time
    with open(inpath, 'rb') as inf:
        csvreader = csv.reader(inf,delimiter=',', quotechar='"')
        
        #Read and print header
        header = inf.readline().split(',')

        #Read and process other lines
        for row in csvreader:
            prep_line(header,row)
            

#process one line.  Produces a dict with key:header, value: data   
def prep_line(header,row):
    row[0] = row[0].replace('"', '').strip()
    line_dict = dict(zip(header,row))
    finished_line = join_essays_to_labels(line_dict['_projectid'],line_dict['essay'])
    write_line(my_outpath, finished_line)
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
    labeled_df['full_essay'][labeled_df['_projectid']==pid] = essay
    if len(labeled_df['full_essay'][labeled_df['_projectid']==pid])>0:
        print "got a match!" + pid
    return labeled_df[labeled_df['_projectid']==pid]

prep_data(full_essays_path,my_outpath)

