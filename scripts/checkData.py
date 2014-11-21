import pandas as pd
from sys import argv
from DataLoading import *

script, second = argv
data = '../data/' + second

freqs = {'t':0, 'f':0, 1:0, 0:0}

def quickSummary(chunk,chunk_id):
    for i, key_i in enumerate(chunk.got_posted.value_counts().keys()):
        freqs[key_i] = freqs[key_i] + chunk.got_posted.value_counts()[key_i]
    print freqs
    return freqs

chunker(data,quickSummary,10000)
print freqs
