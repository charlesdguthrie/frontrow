from nltk.stem import PorterStemmer


def stemming_function (word_list):

  stemmer = PorterStemmer()
  
  stemmed_word_list = []

  for word in word_list:
    stemmed = stem(word)
    stemmed_word_list.append(stemmed)

  return stemmed_word_list
