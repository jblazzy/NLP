# Naive Bayes Classifier

import nltk
import math
from nltk.corpus import movie_reviews, stopwords

# get our movie reviews from nltk.corpus (reviews stored as tuples (review, class))
documents = [(movie_reviews.raw(fileid), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]
classes = movie_reviews.categories() # [’pos’, ’neg’]

trainingSet = documents[100:900] + documents[1100:1900]
devSet = documents[900:1000] + documents[1900:]
testSet = documents[:100] + documents[1000:1100]

def train(trainingSet, classes):
    # train the data
    n = len(trainingSet)    # total number of docs 
    log_prior = {}  # dictionary to hold log prior for all cases

    fulltext = ""

    # dictionary that holds bigdoc for each class
    bigdoc_dict = {}

    # dictionary that holds number of docs in each class
    num_docs = {}
    for c in classes:
        bigdoc_dict[c] = ""
        num_docs[c] = 0
        log_prior[c] = 0 

    # divides training set into positive and negative reviews
    for d in trainingSet:
        fulltext += d[0] + " "
        c = d[1]
        num_docs[c] += 1
        bigdoc_dict[c] += d[0] + " "
    
    # calculate log priors
    for c in classes:
        n2 = num_docs[c]    # number of docs for class c
        log_prior[c] = math.log(n2/n)
        bigdoc_dict[c] = bigdoc_dict[c].lower().strip(".,;!?:")   # sets text to lowercase
