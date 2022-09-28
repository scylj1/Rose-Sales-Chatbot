# Compare the cosine similarity between inputs and texts in the database

from scipy import spatial
import numpy as np
import random
import json
from common import get_document, tfidf_weighting

class cosine_similarity:
    
    N = 0  # number of documents
    n = []
    pairs = []  # Q & A pairs
    vocabulary = []
    bow = []
    intents = []

    def __init__(self):
        self.pairs = self.load_corpus()
        self.bow, self.vocabulary = self.get_bow(self.pairs)

    '''
    Load question answer corpus

        Argument: None
     
        Returns: A list of questions and answers pairs
    '''

    def load_corpus(self):
        pairs = []

        # load corpus
        data_file = open('data/intents.json').read()
        self.intents = json.loads(data_file)

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                pairs.append([pattern, intent['tag']])
        return pairs

    '''
    Get bag of words

        Argument: A list of questions and answers pairs
     
        Returns: Bag of words and total vocabulary
    '''

    def get_bow(self, pairs):
        documents = []
        for pair in pairs:
            documents.append(get_document(pair[0], stop=False))
        # Create vocabulary
        vocabulary = []
        for document in documents:
            for item in document:
                if item not in vocabulary:
                    vocabulary.append(item)
        self.N = len(documents)
        self.n = np.zeros(len(vocabulary))
        for voc in vocabulary:
            index = vocabulary.index(voc)
            for doc in documents:
                if voc in doc:
                    self.n[index] += 1

        bow = []
        for document in documents:
            vector = np.zeros(len(vocabulary))
            for item in document:
                index = vocabulary.index(item)
                vector[index] += 1
            bow.append(tfidf_weighting(vector, np.array(self.n), self.N))
        return bow, vocabulary

    '''
    Check similarity for userinput with database

        Argument: Input sentence, if eliminate stop word, if use stemming
     
        Returns: A list of similarities
    '''

    def get_similarity_cosine(self, query, stop=True, stemming=True):
        stemmed_query = get_document(query, stop=False)
        vector_query = np.zeros(len(self.vocabulary))
        for stem in stemmed_query:
            try:
                index = self.vocabulary.index(stem)
                vector_query[index] += 1
            except ValueError:
                continue
        vector_query = tfidf_weighting(vector_query, np.array(self.n), self.N)

        similarities = []
        if sum(vector_query) != 0:
            for vector in self.bow:
                similarities.append(
                    1 - spatial.distance.cosine(vector, vector_query))
        else:
            similarities.append(0)
        return similarities

    '''
    Get a random response from that intent

        Argument: A tag
     
        Returns: A random response from that tag
    '''

    def get_response(self, tag):
        
        for intent in self.intents['intents']:
            if(intent['tag'] == tag):
                result = random.choice(intent['responses'])
                break
        return result

    '''
    Get response for uer input

        Argument: Input sentence
     
        Returns: response
    '''
    def talk_response(self, user_input):
        similarities = self.get_similarity_cosine(user_input, stop = False, stemming=False)
        max_similar = max(similarities)
        if (max_similar > 0.8):
            max_index = similarities.index(max_similar)
            result = self.get_response(self.pairs[max_index][1])
        else: 
            result = "Sorry, I do not understand."
        
        return result
