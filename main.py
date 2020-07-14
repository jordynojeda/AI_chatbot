


# Created on Thu Jul  9 20:01:20 2020

# @author: jordynaojeda



import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tflearn
import tensorflow
import random
import json
import pickle

try:
    nltk.download('punkt')
except:
    pass

with open("intents.json") as file:  # Open the intents.json file
   data = json.load(file)           # Assign data to the json file
     
try:
   with open("Data.pickle", "rb") as f:          # Reads the file in as bytes
       words, labels, training, output = pickle.load(f)   # Saves the four variables into the pickle file. Then it loads in the four lists because those are the only ones we need
   
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
        
    for intent in data["intents"]:          # Loops through all of the dictionaries in intents.json
        for pattern in intent["patterns"]: 
             wrds = nltk.word_tokenize(pattern) # Returns a list of all the different words in it
             words.extend(wrds)                 # Adds all the words in intents in the words list
             docs_x.append(wrds)
             docs_y.append(intent['tag'])            
              
             if intent["tag"] not in labels:    # Adds all the tags to the labels list
                 labels.append(intent["tag"])
                   
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]   # Finds out how many words it has seen already. Also doesn't include question marks 
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
        
    with open("Data.pickle", "wb") as f:          # Writes the file in as bytes
           pickle.dump((words, labels, training, output),f)  # Writes all these variables into a pickle file and save it

# AI aspect of this project

tensorflow.reset_default_graph()                 # Gets ride of all old setting and resets the graph

net = tflearn.input_data(shape = [None, len(training[0])])   # Gets the input length
net = tflearn.fully_connected(net, 8) # start at the input data then add Eight neurons for the hidden layer
net = tflearn.fully_connected(net, 8) # start at the input data then add Eight neurons for the hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax") # This allows us to get probabilities for each output "softmax" gives a probability for each neuron for this layer
net = tflearn.regression(net)

model = tflearn.DNN(net)       #Type of neural network. Takes the network in we made and uses it

try:
    model.load("model.tflearn")
  
except: 
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)  
    model.save("model.tflearn")           


def bag_of_words(s, words):       # Function to make sentences
    bag = [0 for _ in range(len(words))]  # Stores the words
    
    s_words = nltk.word_tokenize(s) #List of tokenized words
    s_words = [stemmer.stem(word.lower()) for word in s_words] #Stems the words
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:                  # Means that the current word you are looking at in your words list is equal to the one in the sentence
                bag[i] = 1
                
            
    return numpy.array(bag)              # Turns the bag of words into a numpy array

def chat():
    print("Start taking with the chatbot!\n")
    print("Type the words ( quit ) to stop.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results =  model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)                # Finds the greatest number in results
        tag = labels[results_index]                        # Labels stores the labels. So by doing this it will give us the label it thinks it is
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    
            print(random.choice(responses))
        else:
            print("I didn't get that, try again")
            
         
chat()
     
 
















           

