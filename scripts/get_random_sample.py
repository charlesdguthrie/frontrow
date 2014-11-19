#get_random_sample.py
from sys import argv
import pandas as pd
import numpy as np
import csv
import util

script, second = argv
outpath = '../data/' + second

essays_path = "../data/full_labeled_essays.csv"

chunksize = 1000
out_columns = ['got_posted','_projectid','_teacherid','title','essay','need_statement','created_date']

#initialize csv file
with open(outpath, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(out_columns)

def sample(chunk,chunk_id):
	rand = np.random.randint(chunk.shape[0], size=chunk.shape[0]/100)
	subsample = chunk.iloc[rand]
	util.write_chunk(outpath, subsample)

util.chunker(1000,essays_path,sample)