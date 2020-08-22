#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:41:37 2020

@author: aishit
"""


from topic_modelling import topic_model
from cleansing import clean_data

data_path = 'data.json'
# clean = clean_data(data_path)
# clean.remove_empty_data()
# clean.divide_sessions()
# clean.Cleansing()

topic_m = topic_model('clean_sessions.npy')
topic_m.Topic()

