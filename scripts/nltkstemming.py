'''this is an oveview of stemming and lemmatization'''

import nltk

'''
For grammatical reasons, documents are going to use different forms of a word, such as organize, organizes, and organizing. Additionally, there are families of derivationally related words with similar meanings, such as democracy, democratic, and democratization. In many situations, it seems as if it would be useful for a search for one of these words to return documents that contain another word in the set.

The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common baseform. 

For instance:
am, are, is -> be 
car, cars, car's, cars' -> car

The result of this mapping of text will be something like:
the boy's cars are different colors -> the boy car be differ color

However, the two words differ in their flavor. Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma . If confronted with the token saw, stemming might return just s, whereas lemmatization would attempt to return either see or saw depending on whether the use of the token was as a verb or a noun. The two may also differin that stemming most commonly collapses derivationally related words, whereas lemmatization commonly only collapses the different inflectional forms of a lemma. Linguistic processing for stemming or lemmatization is often done by an additional plug-in component to the indexing process, and a number of such components exist, both commercial and open-source.

Stemmers are much simpler and faster than lemmatizers, for many applications the results of stemmers are good enough.

3 common stemmer and lemmatizers:
1. PorterStemmer
2. WordNet Lemmatizer
3. Snowball

GOOD RESOURCES:
check out the stanford nlk: http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
stemmer and lemmatizer demo: http://text-processing.com/demo/stem/
NLTK Stemmers: http://www.nltk.org/api/nltk.stem.html
'''


'''
PorterStemmer

http://www.tartarus.org/~martin/PorterStemmer/

'''

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem('cooking') #cook
stemmer.stem('cookery') #cookeri
stemmer.stem('books') #book
stemmer.stem('said') #said
stemmer.stem('feet') #feet


'''
WordNet:

WordNet is an NLTK corpus reader that includes 

for wordnet sourcecode, refer to http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

to use the wordnet lemmatizer, each word must be tagged with its pos. The default is noun
'''

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
'''to tag the pos of string. the parameter can be a list of string'''
tagged = nltk.pos_tag("dying")

'''this function converts the nltk pos tag to wordnet tag'''
def get_wordnet_pos(tagged):
if tagged.startswith('J'):
   ....:         return wordnet.ADJ
   ....:     elif tagged.startswith('V'):
   ....:         return wordnet.VERB
   ....:     elif tagged.startswith('N'):
   ....:         return wordnet.NOUN
   ....:     elif tagged.startswith('R'):
   ....:         return wordnet.ADV
   ....:     else:
   ....:         return ""

'''the following scripts lemmatizes words using wordnet'''
lmt = WordNetLemmatizer()
lmt.lemmatize('dying', wordnet.VERB) #die
lmt.lemmatize('cooking') #cooking
lmt.lemmatize('cooking', wordnet.VERB) #cook
lmt.lemmatize('cookbooks', wordnet.NOUN) #cookbook
lmt.lemmatize('brought', wordnet.VERB) #bring
lmt.lemmatize('brought') #brought

'''
Snowball

http://www.nltk.org/_modules/nltk/stem/snowball.html
http://snowball.tartarus.org/algorithms/english/stemmer.html

Snoball supports a number of languages:
danish dutch english finnish french german hungarian italian
norwegian porter portuguese romanian russian spanish swedish
'''

from nltk.stem.snowball import SnoballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("running") # run

'''decide to not stem stopwords'''
stemmer = SnowballStemmer("english", ignore_stopwords = True)
stemmer.stem("having") #having