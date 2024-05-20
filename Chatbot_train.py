#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries
# 
# pickle: This library is used for serializing and deserializing Python objects. Here, it's used to save and load Python objects like lists and dictionaries.
# 
# nltk: Natural Language Toolkit (NLTK) is a library used for natural language processing (NLP) tasks like tokenization, lemmatization, stemming, etc.
# 
# numpy: This library is used for numerical computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.
# 
# keras: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It's used here for building and training the neural network model.

# In[4]:


import pickle
get_ipython().system('pip install nltk')


# In[5]:


get_ipython().system('pip install tensorflow keras nltk')


# In[7]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


# ## Tokenization and Lemmatization
# 
# The script reads a JSON file containing intents for the chatbot. Each intent consists of patterns, which are sentences or phrases users might type in.
# It loops through each intent and its associated patterns.
# 
# For each pattern, it tokenizes the sentence into individual words using NLTK's word_tokenize() function. Tokenization involves breaking a text into individual words or tokens.
# 
# It adds these words to a list called words.
# 
# It also creates a list called documents, which contains tuples of words and their corresponding intents. This will be used later for training the model.
# Additionally, it maintains a list called classes, which stores all the unique intents.

# In[8]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


# ## Data Preparation:
# 
# After collecting all the words, it performs lemmatization on them. Lemmatization is the process of reducing words to their base or dictionary form (lemma).
# 
# It converts all words to lowercase to ensure consistency.
# Duplicate words are removed from the list, resulting in a list of unique words.
# 
# The script saves the unique words and intents using the pickle module for future use.

# In[16]:


import json
import nltk

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Open the JSON file with UTF-8 encoding
with open('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/Chatbot/chat_json.json', encoding='utf-8') as intents_file:
    intents_data = json.load(intents_file)

# Loop through intents and their patterns
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)        
        # Add documents in the corpus
        documents.append((word, intent['intent']))  # Use 'intent' instead of 'tag'
        # Add to our classes list
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

print(documents)
print(classes)
print(words)


# ## Bag of Words Creation:
# 
# The script initializes an empty list called training to hold training data.
# It iterates through each document in documents.
# 
# For each document, it creates a bag of words representation. This is a binary vector indicating which words from the vocabulary are present in the current document.
# 
# It creates an output row, which is a list of zeros with a one at the index corresponding to the intent of the document.
# 
# The document's bag of words and output row are appended to the training list.
# The training data is shuffled randomly to prevent the model from learning any order dependencies.
# 

# In[29]:


# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/Chatbot/words.pkl','wb'))
pickle.dump(classes,open('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/Chatbot/classes.pkl','wb'))


# In[22]:


# Create output labels
output_empty = [0] * len(classes)

# Initialize training data
training = []

# Create bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


# In[24]:


# Shuffle the training data
random.shuffle(training)

# Separate features and labels
train_x = []
train_y = []

for features, label in training:
    train_x.append(features)
    train_y.append(label)

# Convert lists to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)


# ## Model Creation:
# 
# The neural network model is defined using Keras Sequential API. This API allows stacking of layers sequentially.
# 
# The model consists of an input layer, two hidden layers, and an output layer.
# The input layer has neurons equal to the length of the bag of words representation.
# 
# Two hidden layers with ReLU activation are added to introduce non-linearity.
# Dropout layers are added after each hidden layer to prevent overfitting by randomly setting a fraction of input units to zero during training.
# 

# In[25]:


# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# ## Model Compilation and Training:
# 
# The model is compiled with categorical cross-entropy loss function, which is suitable for multi-class classification problems.
#  
# The Adam optimizer is used for optimization, which is an extension to stochastic gradient descent (SGD).
# 
# The model is then trained on the training data for a specified number of epochs (iterations over the entire dataset) and with a specified batch size (number of samples per gradient update).
# 
# ## Model Saving:
# 
# Once trained, the model is saved to a file using the save() method provided by Keras.
# 
# Additionally, other necessary files like words, classes, and training data are saved using pickle for later use.

# In[27]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Save the trained model and other necessary files
model.save('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/Chatbot/chatbot_model.h5')
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data.pkl', 'wb'))


# In[28]:


print("model created")

