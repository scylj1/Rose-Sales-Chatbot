'''

Tests for different methods

'''
from cosine_similarity import cosine_similarity
from gru import gru
import json
import os 
from glove import glove
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings
from sequence import sequence
import time

def cosine_test():

    accurate = 0
    total = 0
    sum_time = 0
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)
    c = cosine_similarity()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            start = time.time()
            similarities = c.get_similarity_cosine(pattern, stop = False, stemming=False)
            max_similar = max(similarities)           
            max_index = similarities.index(max_similar)
            tag = c.pairs[max_index][1]
            end = time.time()
            sum_time = sum_time + float(end - start)
            if tag == intent["tag"]:
                accurate += 1
            total += 1
    print("Accuracy of cosine similarity method on training data ")
    print(accurate/total)
    print("Time cost of cosine similarity method on each input ")
    print(sum_time/total)

def sequence_test():
    accurate = 0
    total = 0
    sum_time = 0
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)
    im = sequence()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            start = time.time()
            im.predict = im.predict_class(pattern, im.model)
            end = time.time()
            sum_time = sum_time + float(end - start)
            if im.predict != []:
                tag = (im.predict)[0]['intent']
                if tag == intent["tag"]:
                    accurate += 1
            total += 1
    print("Accuracy of sequential method on training data ")
    print(accurate/total)
    print("Time cost of sequential method on each input ")
    print(sum_time/total)


def gru_test():
    accurate = 0
    total = 0
    sum_time = 0
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)
    g = gru()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            start = time.time()
            g.predict = g.predict_class(pattern, g.model)
            end = time.time()
            sum_time = sum_time + float(end - start)
            if g.predict != []:
                tag = g.predict
                if tag == intent["tag"]:
                    accurate += 1
            total += 1
    print("Accuracy of gru method on training data ")
    print(accurate/total)
    print("Time cost of gru method on each input ")
    print(sum_time/total)

def glove_test():
    accurate = 0
    total = 0
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)
    w = glove()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w.predict = w.predict_class(pattern, w.model)
            if w.predict != []:
                tag = w.predict
                if tag == intent["tag"]:
                    accurate += 1
            total += 1
    print("Accuracy of gru method on training data ")
    print(accurate/total)
    
    
if __name__ == '__main__':
    cosine_test()   
    sequence_test() 
    gru_test()
    #glove_test()