#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:03:29 2020

@author: aishit
"""


import json
import dateutil.parser as dparser
import numpy as np
import re
import nltk
import nltk.corpus
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
# import en_core_web_sm

        
        
class clean_data:
    def __init__(self, path):
        self.data = json.loads(open(path).read())
        nltk.download('stopwords')
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        
    def remove_empty_data(self):
        
        non_empty_sessions = []
        for i in range(len(self.data)):
            if self.data[i]['time'] == [] or self.data[i]['content'] == []:
                continue
            else:
                non_empty_sessions.append(self.data[i])
                
        self.non_empty_data = non_empty_sessions
    
    def same_session(self, search_1, search_2):
        parse_1 = dparser.parse(search_1['time'][1], fuzzy=True)
        parse_2 = dparser.parse(search_2['time'][1], fuzzy=True)
        
        delta = parse_1 - parse_2
        
        if delta.total_seconds()/60.0 <= 30.0:
            return True
        else:
            return False
    
    def divide_sessions(self):
        
        self.sessions = []
        session = [self.non_empty_data[0]]
        
        for i in range(1, len(self.non_empty_data)):
            search_1 = self.non_empty_data[i-1]
            search_2 = self.non_empty_data[i]
            
            if search_1['time'] == []:
                continue
            
            if self.same_session(search_1, search_2):
                session.append(search_2)
            else:
                self.sessions.append(session)
                session = [self.non_empty_data[i]]
        
        print('len(sessions)', len(self.sessions))
        np.save('sessions.npy', self.sessions)
        
    def Cleansing(self):
        
        clean_sessions = []
        
        stop = stopwords.words('english')
        
        nlp = en_core_web_sm.load()
        
        for session in self.sessions:
            clean_session = []
            for i in range(len(session)):
                
                # lowercase
                session[i]['content'][0] = session[i]['content'][0].lower()
                # remove symbols and urls
                session[i]['content'][0] = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", session[i]['content'][0])
                # remove numbers
                session[i]['content'][0] = re.sub(r"\d+", "", session[i]['content'][0])
                
                session[i]['content'][0] = ' '.join([word for word in session[i]['content'][0].split() if word not in (stop)])
                
                if session[i]['content'][0] == '':
                    continue
                else:
                    doc = nlp(session[i]['content'][0])
                    
                    doc_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                    
                    session[i]['Noun phrases'] = [chunk.text for chunk in doc.noun_chunks]
                    session[i]['Verbs'] = doc_verbs
                    session[i]['Entities'] = [[entity.text, entity.label_] for entity in doc.ents]
                    clean_session.append(session[i])
                    
            
            if clean_session != []:
                clean_sessions.append(clean_session)
            
        self.clean_sessions = clean_sessions
        np.save('clean_sessions.npy', self.clean_sessions)
            
    def load_sessions(self):
        print('load sessions')
        self.sessions = np.load('sessions.npy', allow_pickle=True)
        self.clean_sessions = np.load('clean_sessions.npy', allow_pickle=True)
        # print('len(sessions)', len(self.sessions))
        
                
# d = clean_data()
# d.remove_empty_data()
# d.divide_sessions()
# d.load_sessions()
# session_number = 100
# print(d.sessions[session_number])
# print('\n')
# d.Cleansing()
# print(d.sessions[session_number])
# print(d.clean_sessions[session_number])