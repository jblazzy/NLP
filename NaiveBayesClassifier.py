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


def test(testDoc, logPrior, logLikelihood, classes, vocab):
    sums = {}
    
    words = nltk.word_tokenize(testDoc)
    stop_words = set(stopwords.words('english')) 

    for c in classes:
        sums[c] = logPrior[c]
        for w in words:
            # ignores words not in vocab from training & stop words
            if (w in logLikelihood[c]) and (w not in stop_words):
                sums[c] += logLikelihood[c][w]

    # returns argmax of sum[c]
    max_log = 0
    argmax = ""
    for c in classes:
        if abs(sums[c]) > abs(max_log):
            max_log = sums[c]
            argmax = c
    return argmax


def testCorpus(testSet, logPrior, logLikelihood, classes, vocab):
    # tests the classifier for all documents in the test set
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # don't know how this can be generalized for other classes
    for doc in testSet:
        result = test(doc[0], logPrior, logLikelihood, classes, vocab)
        if result == "pos" and doc[1] == "pos":
            tp += 1
        if result == "pos" and doc[1] == "neg":
            fp += 1
        if result == "neg" and doc[1] == "pos":
            fn += 1
        if result == "neg" and doc[1] == "neg":
            tn += 1

    # we tried here...
    # for doc in testSet:
    #     for c in classes:
    #         result = test(doc[0], logPrior, logLikelihood, classes, vocab)
    #         if result == doc[1]:
    #             tp += 1
    #         if result != doc[1]:
    #             tn += 1            

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (precision, recall)


results = train(trainingSet, classes)

logPrior = results[0]
logLikelihood = results[1]
vocab = results[2]

# tweaking
results = testCorpus(testSet, logPrior, logLikelihood, classes, vocab)

print("Precision: " + str(results[0]) + ", Recall: " + str(results[1]))