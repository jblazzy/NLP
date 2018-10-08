# Naive Bayes Classifier

import nltk
import math
from nltk.corpus import movie_reviews, stopwords

# get our movie reviews from the nltk.corpus import
documents = [(movie_reviews.raw(fileid), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]
classes = movie_reviews.categories() # [’pos’, ’neg’]

trainingSet = documents[100:900] + documents[1100:1900]
devSet = documents[900:1000] + documents[1900:]
testSet = documents[:100] + documents[1000:1100]
