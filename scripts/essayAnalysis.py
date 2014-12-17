
tw = pd.read_csv('../data/topwords_test.csv', index_col=0)
tw.columns = ['word','freq_rej','freq_app']

def minFreq(df,myMin):
    return df[(df['freq_rej']>myMin) & (df['freq_app']>myMin)]

tw = minFreq(tw,0)

def distinctiveWords(df,myMin):
    df2 = minFreq(df,myMin)
    df2['ratio_rej'] = df2.freq_rej/df2.freq_app
    df2['ratio_app'] = df2.freq_app/df2.freq_rej
    df2 = df2.sort(columns='ratio_rej',ascending=False)
    print "Rejected Words"
    print df2[:10]
    print ""
    print "Approved Words"
    df2 = df2.sort(columns='ratio_app',ascending=False)
    print df2[:10]


distinctiveWords(tw,100)
distinctiveWords(tw,300)
distinctiveWords(tw,10)