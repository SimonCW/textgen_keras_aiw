
# coding: utf-8

# In[6]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import sys


# Import ASCII text of the books "Alice in Wonderland" (downloaded from project Gutenberg) and convert all characters to lowercase so that the model does not treat capitalized words as new words.

# In[7]:


# load text and convert to lowercase
filename = "./input/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# For local testing, I decrease the text size dramatically to the first 1000 characters.

# In[8]:


# decrease input text size drastically for local testing
#raw_text = raw_text[0:1000]


# So that the Neural Network can use the input, each unique character is mapped to an integer value.

# In[9]:


# create mapping of unique charst to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))


# Summary Statistics of the data set

# In[10]:


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", str(n_chars))
print("Total distinct characters: ", str(n_vocab))


# A training pattern are the firtst 100 characters, the ground-truth "label" for this first training pattern then is the 101st character. This window of 100 characters then gets slided character by character.
# 
# For illustration purposes assume we would take a sequence of 5 characters, then the first training sample would be `chapt` -> `e` and the second training sample would be `hapte` -> `r`. 
# 
# The characters get converted to integers using the lookup dictionary created before.

# In[11]:


# Prepare dataset
seq_length = 100
trainX = []
trainY = []
for i in range(0, n_chars-seq_length, 1):
    seq_in = raw_text[i : i + seq_length]   # in 1st iteration contains first 100 chars
    seq_out = raw_text[i + seq_length]   # in 1st iteration contains 101st char
    trainX.append([char_to_int[char] for char in seq_in])   # char is the character as string, char_to_int[char] gives the int value
    trainX
    trainY.append(char_to_int[seq_out])
n_patterns = len(trainX)
print("Total # of Patterns: " + str(n_patterns))


# To train the network we need to transform the data further. 
# 1. Reshape the training data to the form [samples, time steps, features].
# 2. Rescale the integers to the range 0-1 to better train the network if a sigmoid function is used (stay in approx linear part of the sigmoid). 
# 3. One-hot encode trainY so that each character in y is represented by a vector of 45 (number of distinct characters) values. The character "n" (inter value 32) is then represented by a vector of zeros except for one "1" in column 32.

# In[12]:


# reshape X to [samples, time steps, features]
X = np.reshape(trainX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode y
y = np_utils.to_categorical(trainY)


# I define a LSTM model with two hidden layers and 256 memory units. Dropout is used with a probability of 20%. The output layer is a Dense layer using the softmax activation function. This outputs a probability prediction for each character. These are the parameters used in the tutorial. They are not very well tuned but merely a starting point.

# In[13]:


model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam")


# Next, I save the model weights after each epoch so that I can use the weights that produced the model with the lowest error for prediction afterwards. I also save the tensorboard callbacks to be able to see 

# In[14]:


# define checkpoints
filepath = "./checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = "loss", verbose = 1, save_best_only = True, mode = "min")
tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks_list = [checkpoint, tensorboard_cb]


# In[15]:


model.fit(X, y, epochs = 60, batch_size = 64, callbacks = callbacks_list)

