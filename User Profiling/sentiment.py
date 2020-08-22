#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:29:53 2020

@author: aishit
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros

class Sentiment:
    def load_data(self):
        self.movie_reviews = pd.read_csv("IMDB Dataset/IMDB Dataset.csv")
        self.glove_file = open('IMDB Dataset/glove.6B.50d.txt', encoding="utf8")

    def data_visualize(self):
        sns.countplot(x='sentiment', data=self.movie_reviews)

    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)
    
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
    
        return sentence

    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)
    
    def prepare_data(self):
        X = []
        sentences = list(self.movie_reviews['review'])
        for sen in sentences:
            X.append(self.preprocess_text(sen))
            
        y = self.movie_reviews['sentiment']
        
        y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        
        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1
        
        maxlen = 100
        
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        
        embeddings_dictionary = dict()
        glove_file = self.glove_file
        
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary [word] = vector_dimensions
            
        glove_file.close()
        
        embedding_matrix = zeros((vocab_size, 50))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.maxlen = maxlen
        
    def create_model(self):
        
        model = Sequential()
        embedding_layer = Embedding(self.vocab_size, 50, weights=[self.embedding_matrix], input_length=self.maxlen , trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(128))
        
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        
        print(model.summary())
        
        self.model = model

    def fit_model(self):
        model = self.model        
        self.history = model.fit(self.X_train, self.y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)        
        self.score = model.evaluate(self.X_test, self.y_test, verbose=1)
        
        model.save_weights('sentiment_model_weights.h5')

        print("Test Score:", self.score[0])
        print("Test Accuracy:", self.score[1])
        
    def model_graphs(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()
        
obj = Sentiment()
obj.load_data()
# obj.data_visualize()
obj.prepare_data()
obj.create_model()
obj.fit_model()
obj.model_graphs()