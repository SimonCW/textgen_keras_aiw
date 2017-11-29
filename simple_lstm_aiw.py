
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys


# Import ASCII text of the books "Alice in Wonderland" (downloaded from project Gutenberg) and convert all characters to lowercase so that the model does not treat capitalized words as new words.

# In[2]:


# load text and convert to lowercase
filename = "./input/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# So that the Neural Network can use the input, each unique character is mapped to an integer value.

# In[3]:


# create mapping of unique charst to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))


# Summary Statistics of the data set

# In[4]:


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", str(n_chars))
print("Total distinct characters: ", str(n_vocab))


# A training pattern are the firtst 100 characters, the ground-truth "label" for this first training pattern then is the 101st character. This window of 100 characters then gets slided character by character.
# 
# For illustration purposes assume we would take a sequence of 5 characters, then the first training sample would be `chapt` -> `e` and the second training sample would be `hapte` -> `r`. 
# 
# The characters get converted to integers using the lookup dictionary created before.

# In[5]:


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

# In[6]:


# reshape X to [samples, time steps, features]
X = np.reshape(trainX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode y
y = np_utils.to_categorical(trainY)


# I define a LSTM model with one hidden layer and 256 memory units. Dropout is used with a probability of 20%. The output layer is a Dense layer using the softmax activation function. This outputs a probability prediction for each character. These are the parameters used in the tutorial. They are not very well tuned but merely a starting point.

# In[7]:


model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam")


# Next, I save the model weights after each epoch so that I can use the weights that produced the model with the lowest error for prediction afterwards. 

# In[10]:


# define checkpoints
filepath = "./output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = "loss", verbose = 1, save_best_only = True, mode = "min") 
callbacks_list = [checkpoint]


# In[11]:


model.fit(X, y, epochs = 4, batch_size = 128, callbacks = callbacks_list)


# ## Generating Text with the saved weights
#     

# In[8]:


# load weights
filename = "weights-improvement-04-2.5865.hdf5"
model.load_weights(filename)
model.compile(loss = "categorical_crossentropy", optimizer = "adam")


# Creating a reverse mapping from integer to character to convert the predicted integers to characters.

# In[9]:


int_to_char = dict((i, c) for i, c in enumerate(chars))


# -

# In[19]:


# return a random integer between 0 and the number of different patterns in the training data
start = np.random.randint(0, len(trainX)-1)
# pick the random pattern
pattern = trainX[start]
print("Seed: ")
# print the random pattern by converting the integers to characater
print("\"", "".join([int_to_char[value] for value in pattern]), "\"")


# Printing 1000 characters of generated text starting with the random pattern above:

# In[20]:


# initialize empty list for the result_output
result_output = []
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # prediction contains the probability for each character (0-45) for the given input pattern x
    prediction = model.predict(x, verbose = 0)
    # index contains the index where the prediction is highest
    index = np.argmax(prediction)
    # the predicted character
    result = int_to_char[index]
    # the input sequence 
    seq_in = [int_to_char[value] for value in pattern]
    # write to console
    sys.stdout.write(result)
    # append predicted index to the result_output list
    result_output.append(index)
    # append predicted index to the pattern
    pattern.append(index)
    # new pattern is the old pattern with the first character cut away and the new prediction appended to the end. this new pattern is the input for the next iteration
    pattern = pattern[1:len(pattern)]


# Print the output and wirte to txt

# In[29]:


with open("./output/prediction.txt", "w") as f:
    f.write("".join([int_to_char[value] for value in result_output]))
    print("done")

