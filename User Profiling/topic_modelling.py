#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:53:33 2020

@author: aishit
"""
from gensim import corpora
import numpy as np
import pandas as pd
import pickle
import gensim
import pyLDAvis.gensim

class topic_model:
    
    def __init__(self, clean_sessions_path):
        self.NUM_TOPICS = 5
        self.sessions = np.load(clean_sessions_path, allow_pickle=True)
        
    def Topic(self):
        session_topics = []
        i=0
        complete_corpus = []
        complete_dictionary = []
        complete_text_data = []
        for session in self.sessions:
            print(i, end=' ')
            i+=1
            text_data = []
            
            for search in session:
                text_data.append(search['content'][0].split())
            
            dictionary = corpora.Dictionary(text_data)
            corpus = [dictionary.doc2bow(text) for text in text_data]
            complete_corpus.extend(dictionary.doc2bow(text) for text in text_data)
            complete_text_data.extend(text_data)
            
            # pickle.dump(corpus, open('corpus.pkl', 'wb'))
            # dictionary.save('dictionary.gensim')
            
            # ldamodel = gensim.models.ldamodel.LdaModel.load('model5.gensim')

            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.NUM_TOPICS, id2word=dictionary, passes=15)
            # ldamodel.save('model5.gensim')
            topics = ldamodel.print_topics(num_words=3)
            session_topics.append(topics)
            
            # lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
            # pyLDAvis.show(lda_display)
            # print(lda_display)
            # df = pd.DataFrame(lda_display)
            # df.to_csv('lda_display.csv')
            # break
        complete_dictionary = corpora.Dictionary(complete_text_data)
        complete_ldamodel = gensim.models.ldamodel.LdaModel(complete_corpus, num_topics = self.NUM_TOPICS, id2word=complete_dictionary, passes=15)
        lda_display = pyLDAvis.gensim.prepare(complete_ldamodel, complete_corpus, complete_dictionary, sort_topics=False)
        pyLDAvis.show(lda_display)
        pyLDAvis.save_html(lda_display, 'lda_display.html')
        pyLDAvis.save_json(lda_display, 'lda_display.json')
        # print(lda_display)
            
        # self.session_topics = session_topics
        # np.save('session_topics.npy', self.session_topics)
        
    def load_topics(self):
        self.session_topics = np.load('session_topics.npy', allow_pickle=True)

# t = topic_model()
# t.Topic()
# t.load_topics()

# print(t.session_topics[0])
# print(t.sessions[0])