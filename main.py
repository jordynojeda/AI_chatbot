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
   data = json.load(file)           # Assign data to the json file
   

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:          # Loops through all of the dictionaries in intents.json
   for pattern in intent["patterns"]: 
       wrds = nltk.word_tokenize(pattern) # Returns a list of all the different words in it
       words.extend(wrds)                 # Adds all the words in intents in the words list
       docs_x.append(pattern)
       docs_y.append(intent['tag'])            
      
       if intent["tag"] not in labels:    # Adds all the tags to the labels list
           labels.append(intent["tag"])
           
words = [stemmer.stem(w.lower()) for w in words]   # Finds out how many words it has seen already
words = sorted(list(set(words)))                   # Removes all the duplicate words that have been found 

labels = sorted(labels)                            # Sorts the labels

#Neutral networks only understand numbers. So have to make a bag of words

training = []
output = []

out_empty = [0 for _ in range(len(labels))] 

for x, doc in enumerate(docs_x):                 # Making the bag of words
    bag = []
    
    wrds = [stemmer.stem(w) for w in doc]        # Stems the words
    
    for w in words:                              # Loops through the words
        if w in wrds:
            bag.append(1)                        # If the word is in wrds then add a 1 to the bag list
        else:
            bag.append(0)                        # Else  add a 0 to the bag list if the word isn't in words
            
            
    output_row = out_empty[:]                    # Look through the labels list and find the tag
    output_row[labels.index(docs_y[x])] = 1      # Finds the tag and sets the value to one
    
    training.append(bag)                         # Training list with bunch a bag of words
    output.append(output_row)                    
    
training = numpy.array(training)                 # Turned into numpy arrays for tflearn
output = numpy.array(output)                     # Turned into numpy arrays for tflearn
     
 
















           

