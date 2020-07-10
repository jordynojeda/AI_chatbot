#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:01:20 2020

@author: jordynaojeda
"""


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:  # Open the intents.json file
   data = json.load(file)          # Assign data to the json file

words = []
labels = []
docs = []

for intent in data["intents"]:          # Loops through all of the dictionaries in intents.json
   for pattern in intent["patterns"]: 
       wrds = nltk.word_tokenize(pattern) # Returns a list of all the different words in it
       words.extend(wrds)                 # Adds all the words in intents in the words list
       docs.append(pattern)              
      
       if intent["tag"] not in labels:    # Adds all the tags to the labels list
           labels.append(intent["tag"])

