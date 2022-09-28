# GRU model for intent matching

import json
import pickle
import keras
import numpy as np
import random
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.layers import Embedding, GRU
from keras.models import Sequential

if __name__ == '__main__':

    classes = []
    labels = []
    texts = []

    # load corpus
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            texts.append(pattern)
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
            
            labels.append(intent['tag'])
            
    # save data
    pickle.dump(texts, open('data/gru/texts.pkl', 'wb'))
    pickle.dump(classes, open('data/gru/classes.pkl', 'wb'))

    # process labels
    y=[]
    x=0
    l1 = labels[0]
    for l in labels:
        
        if l != l1:
            x=x+1
        y.append(x)

        l1 = l
    #print(y)

    # model parameters
    MAX_SEQUENCE_LENGTH = 100 # max sequence length
    EMBEDDING_DIM = 200  # embdding dimension
    VALIDATION_SPLIT = 0  # validation set

    # preprocessing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(y))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print("data:", data)
    print("labels:", labels)
    print(data[-1])

    # set up model
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(GRU(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()

    # structure graph
    keras.utils.vis_utils.plot_model(model, "model/gru.png", show_shapes=True, dpi=1440)

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
                
   # train
    model.fit(data, labels, epochs=50, batch_size=16)
    model.save('model/gru_model')


class gru:
    
    model = []
    classes = []
    texts = []
    predict = []
    intents= []

    def __init__(self):
        self.model = load_model('model/gru_model')
        self.intents = json.loads(open('data/intents.json').read())
        self.classes = pickle.load(open('data/gru/classes.pkl', 'rb'))
        self.texts = pickle.load(open('data/gru/texts.pkl', 'rb'))


    '''
    Predict the intent of user input

        Argument: Input sentence, model
     
        Returns: List of classes
    '''

    def predict_class(self, sentence, model):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences([sentence])
        word_index = tokenizer.word_index

        MAX_SEQUENCE_LENGTH = 100 
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        pre_result = model(data)
        #print(pre_result[0])

        results = []
        i = 0
        for tag in self.classes:
            results.append(float(pre_result[0][i]))
            i=i+1

        max_similar = max(results)
        # print(max_similar)
        if max_similar > 0.7:
            max_index = results.index(max_similar)     
            tag = self.classes[max_index]
        else:
            tag = "none"
        #print(tag)
        return tag

    '''
    Get a random response from that intent

        Argument: A tag
     
        Returns: A random response from that tag
    '''

    def get_response(self, predict, intents_json):
        tag = predict
        if tag == "none":
            return "Sorry, I do not understand."
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
    
