import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
#%% Importing Data
number_of_words = 20000
max_len=100
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=number_of_words)
#%% Padding all sequences to be the same lenght
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen = max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,maxlen = max_len)
#%% Building the RNN
model = tf.keras.Sequential()
#%% Adding Embedded Layer
model.add(tf.keras.layers.Embedding(input_dim=number_of_words,output_dim=128,
                                    input_shape =(X_train.shape[1],)))
#%%Adding the LSTM Layer
model.add(tf.keras.layers.LSTM(128,activation="tanh"))
#%% Adding the output layer
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.summary()
#%%Compiling the model
model.compile(optimizer="rmsprop",loss = "binary_crossentropy",metrics=["accuracy"])
#%% Fitting the Model
model.fit(X_train,y_train,epochs=5)
#%% Evaluating the model
test_loss,test_accuracy = model.evaluate(X_test,y_test)