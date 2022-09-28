'''

Train a 2 layer sequential model for intent matching

'''
import keras
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from common import get_document, tfidf_weighting
import os
import random
from keras.models import load_model
import pydot

if __name__ == '__main__':
    classes = []
    documents = []
    vocabulary = []
    labels = []

    # load corpus
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
            document = get_document(pattern)
            documents.append(document)
            labels.append(intent['tag'])
            for item in document:
                if item not in vocabulary:
                    vocabulary.append(item)

    # get parameters
    N = len(documents)
    n = np.zeros(len(vocabulary))
    for voc in vocabulary:
        index = vocabulary.index(voc)
        for doc in documents:
            if voc in doc:
                n[index] += 1

    # Create bag-of-words input and output
    bow = []
    out_put = []
    y0 = [0] * len(classes)

    for document in documents:
        index = documents.index(document)
        y = list(y0)
        y[classes.index(labels[index])] = 1
        out_put.append(y)

        vector = np.zeros(len(vocabulary))
        for item in document:
            index = vocabulary.index(item)
            vector[index] += 1
        bow.append(tfidf_weighting(vector, n, N))

    # save data
    pickle.dump(vocabulary, open('data/sequence/words.pkl', 'wb'))
    pickle.dump(classes, open('data/sequence/classes.pkl', 'wb'))
    pickle.dump(n, open('data/sequence/n.pkl', 'wb'))
    pickle.dump(N, open('data/sequence/documents.pkl', 'wb'))
    pickle.dump(bow, open('data/sequence/bow.pkl', 'wb'))
    pickle.dump(labels, open('data/sequence/labels.pkl', 'wb'))

    # create train and test sets
    train_x = list(bow)
    train_y = list(out_put)

    # create sequential model - 2 layers
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # structure graph
    keras.utils.vis_utils.plot_model(model, "model/sequential.png", show_shapes=True, dpi=1440)

    # compile model
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    # fit model
    hist = model.fit(np.array(train_x), np.array(train_y),
                     epochs=300, batch_size=5, verbose=1)

    # save model
    model.save('model/sequence_model')


class sequence:

    model = []
    intents = []
    words = []
    classes = []
    n = []
    N = []
    predict = []
    bag_of_words = []
    labels = []

    def __init__(self):
        self.model = load_model('model/sequence_model')
        self.intents = json.loads(open('data/intents.json').read())
        self.words = pickle.load(open('data/sequence/words.pkl', 'rb'))
        self.classes = pickle.load(open('data/sequence/classes.pkl', 'rb'))
        self.n = pickle.load(open('data/sequence/n.pkl', 'rb'))
        self.N = pickle.load(open('data/sequence/documents.pkl', 'rb'))

        self.bag_of_words = pickle.load(open('data/sequence/bow.pkl', 'rb'))
        self.labels = pickle.load(open('data/sequence/labels.pkl', 'rb'))

    '''
    Get bag of words

        Argument: Input sentence and vocabulary
     
        Returns: Bag of words and total vocabulary
    '''

    def get_bow(self, sentence, words):
        # tokenize
        sentence_words = get_document(sentence)
        # bag of words
        vector = np.zeros(len(words))
        for item in sentence_words:
            try:
                index = words.index(item)
                vector[index] += 1
            except ValueError:
                continue
        bag = tfidf_weighting(vector, self.n, self.N)

        return(np.array(bag))

    '''
    Predict the intent of user input

        Argument: Input sentence, model
     
        Returns: List of classes
    '''

    def predict_class(self, sentence, model):
        bow = self.get_bow(sentence, self.words)
        if sum(bow) == 0:
            return []

        res = model(np.array([bow]))[0]

        ERROR_THRESHOLD = 0.8
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append(
                {"intent": self.classes[r[0]], "probability": str(r[1])})

        return return_list

    '''
    Get a random response from that intent

        Argument: A tag, database
     
        Returns: A random response from that tag
    '''

    def get_response(self, predict, intents_json):
        tag = predict[0]['intent']
        intents = intents_json['intents']
        for intent in intents:
            if(intent['tag'] == tag):
                result = random.choice(intent['responses'])
                break
        return result

    '''
    Get response for uer input

        Argument: Input sentence
     
        Returns: response
    '''

    def talk_response(self, userinput):
        self.predict = self.predict_class(userinput, self.model)
        if self.predict != []:
            response = self.get_response(self.predict, self.intents)
            return response
        else:
            return "Sorry, I do not understand"

