import pandas as pd
from sys import argv

script, second = argv
data = '../data/' + second

df=pd.read_csv(data,sep=',', header=0)
print df.got_posted.value_counts()
