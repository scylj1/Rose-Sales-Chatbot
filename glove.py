import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import keras
from nltk.corpus import stopwords
import json
from common import get_document
from keras.preprocessing.sequence import pad_sequences
import random
from keras.models import load_model
from keras.layers import GRU
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.layers import Embedding, GRU
from keras.models import Sequential

if __name__ == '__main__':

    #glove_file = datapath('data/glove/glove.42B.300d.txt')
    tmp_file = get_tmpfile('data/glove/word2vec.42B.300d.txt')
    #_ = glove2word2vec(glove_file, tmp_file)
    glove_model = KeyedVectors.load_word2vec_format(tmp_file)

    data_dim = 300
    glove_model.index_to_key[0]
    stop_words = set(stopwords.words('english'))

    classes = []
    documents = []
    vocabulary = []
    labels = []
    texts = []

    # load corpus
    #file = os.path.abspath("data/intents.json")
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            texts.append(pattern)
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
            document = get_document(pattern, False, False)
            documents.append(document)
            labels.append(intent['tag'])


    # Create bag-of-words input and output
    x_tokens_words = []

    for document in documents:
        index = documents.index(document)

        words = []
        for word in document:
            try:
                words.append(glove_model.key_to_index[word])
            except:
                words.append(0)
        x_tokens_words.append(words)
            
    # save data
    pickle.dump(texts, open('data/glove/texts.pkl', 'wb'))
    pickle.dump(classes, open('data/glove/classes.pkl', 'wb'))

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
    data = pad_sequences(x_tokens_words, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(y))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print("data:", data)
    print("labels:", labels)
    print(data[-1])

    # set up model
    model = Sequential()
    model.add(Embedding(len(glove_model) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(GRU(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()

    # structure graph
    keras.utils.vis_utils.plot_model(model, "model/glove.png", show_shapes=True, dpi=1440)

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
                
   # train
    model.fit(data, labels, epochs=30, batch_size=16)
    model.save('model/glove_model')


class glove:
    
    model = []
    classes = []
    texts = []
    predict = []
    intents= []

    def __init__(self):
        self.model = load_model('model/glove_model')
        self.intents = json.loads(open('data/intents.json').read())
        self.classes = pickle.load(open('data/gru/classes.pkl', 'rb'))
        self.texts = pickle.load(open('data/gru/texts.pkl', 'rb'))


    '''
    Predict the intent of user input

        Argument: Input sentence, model
     
        Returns: List of classes
    '''

    def predict_class(self, sentence, model):
        
        document = get_document(sentence, False, False)
        tmp_file = get_tmpfile('data/glove/word2vec.42B.300d.txt')

        glove_model = KeyedVectors.load_word2vec_format(tmp_file)
        glove_model.index_to_key[0]

        x_tokens_words = []
        words = []
        for word in document:
            try:
                words.append(glove_model.key_to_index[word])
            except:
                words.append(0)
        x_tokens_words.append(words)

        MAX_SEQUENCE_LENGTH = 100  
        data = pad_sequences(x_tokens_words, maxlen=MAX_SEQUENCE_LENGTH)
        pre_result = model(data)
        #print(pre_result[0])

        results = []
        i = 0
        for tag in self.classes:
            results.append(float(pre_result[0][i]))
            i=i+1

        max_similar = max(results)
        max_index = results.index(max_similar)     

        tag = self.classes[max_index]
        #print(tag)
        return tag

    '''
    Get a random response from that intent

        Argument: A tag, database
     
        Returns: A random response from that tag
    '''

    def get_response(self, predict, intents_json):
        tag = predict
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

