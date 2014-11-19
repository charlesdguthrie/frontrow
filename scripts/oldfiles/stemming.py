from nltk.stem import PorterStemmer


def stemming(word_list):

  stemmer = PorterStemmer()
  
  stemmed_word_list = []

  for word in word_list:
    stemmed = stemmer.stem(word)
    stemmed_word_list.append(stemmed)

  return stemmed_word_list
