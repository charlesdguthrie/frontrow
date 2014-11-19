from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

def lemmatizing(word_list):
  '''
  this function lemmatizes a list of words
  '''
  lemmatized = []
  lmt = WordNetLemmatizer()

  tagged = nltk.pos_tag(word_list)
  
  i = 0
  while i < len(tagged):
    wordnet_tagged = get_wordnet_pos(tagged[i][1])
    if wordnet_tagged == "":
      new_word = lmt.lemmatize(word_list[i])
    else:
      new_word = lmt.lemmatize(word_list[i],wordnet_tagged)
    lemmatized.append(new_word)
    i += 1

  return lemmatized

def get_wordnet_pos(tagged):
  '''
  this function converts the nltk pos tags to wordnet pos tags
  '''
  
  if tagged.startswith('J'):
    return wordnet.ADJ
  elif tagged.startswith('V'):
    return wordnet.VERB
  elif tagged.startswith('N'):
    return wordnet.NOUN
  elif tagged.startswith('R'):
    return wordnet.ADV
  else:
    return ""
