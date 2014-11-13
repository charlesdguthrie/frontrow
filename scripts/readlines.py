import pandas as pd

homedir = '/Users/205341/Dropbox/NYU/Intro DS/DonorsChoose/'
my_inpath = homedir + 'essays_and_labels_10.csv'
my_outpath = homedir + 'structured_data.csv'

def prep_data(inpath,outpath):
    import csv

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
    line_dict = dict(zip(header,row))
    print line_dict['title']
    '''
    -----------------------------------
    Insert text processing modules here, eg:
    punct_free = remove_punctuation(line_dict['essay'])
    stopwords_free = remove_stopwords(punct_free)
    stemmed = stem(stopwords_free)
    -----------------------------------
    '''
    
#write processed output one line at a time.  
def write_line(outpath,outline):
    with open(outpath, 'a') as file:
        file.write(outline)


prep_data(my_inpath, my_outpath)
    