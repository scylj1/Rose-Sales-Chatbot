# Common functions for processing texts

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from math import log10
from nltk.stem import SnowballStemmer

'''
Tokenize, remove stopwords and get stemming

    Argument: string: input sentence
                stop: decide whether remove stopwords
            stemming: decide whether do stemming
     
    Returns: processed vector
'''

def get_document(string, stop=True, stemming=True):

    # Tokenise and remove punctuation
    tok_document = word_tokenize(string)

    if stop is True:
        # Remove stop words and normalise casing
        english_stopwords = stopwords.words('english')
        document = [word.lower() for word in tok_document if word.lower()
                    not in english_stopwords]
    else:
        document = [word.lower() for word in tok_document]

    # print(documents)
    if stemming is True:
        # Stemming
        sb_stemmer = SnowballStemmer('english')
        stemmed_document = [sb_stemmer.stem(word) for word in document]
        document = stemmed_document

    return document


'''
Tf-idf weighting method

    Argument: vector: original vector
                   n: number of documents contain the word
                   N: number of total documents
     
    Returns: processed vector
'''

def tfidf_weighting(vector, n, N):
    tfidf_vector = []
    i = 0
    for frequency in vector:
        n = np.array(n)
        if n[i] != 0:
            tfidf_vector.append(log10(1+frequency)*log10(N/n[i]))
        else:
            tfidf_vector.append(0)
        i += 1
    return np.array(tfidf_vector)
