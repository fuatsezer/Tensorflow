import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.datasets import fashion_mnist
#%% Import the data
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
#%% Normalizing the image
X_train = X_train /255.0
X_test = X_test / 255.0
#%%Reshape the datasets
X_train = X_train.reshape(-1,28*28)
print(X_train.shape)
X_test = X_test.reshape(-1,28*28)
print(X_test.shape)
#%% Building ANN
# Defining the model
model = tf.keras.models.Sequential()
#Adding first fully-connected hidden layer
# Layer hyper parameters:
    #number of neurons:128
    #activation function:relu
    #input shape:(784,)
model.add(tf.keras.layers.Dense(128,activation="relu",input_shape=(784,)))
# Adding a second layer with Dropout
# Dropout, bir katmandaki nöronları rastgele sıfıra ayarladığımız bir Regulizasyon tekniğidir
# ve bu şekilde bu nöronları eğitirken güncellenmez. Nöronların belirli bir yüzdesi 
# güncellenmeyeceğinden, tüm eğitim süreci uzundur ve fazla uyum için daha az şansımız var.
model.add(tf.keras.layers.Dropout(0.2))
# Adding the output layer
# number of classes(10 in fashion mnist)
# activation: softmax
model.add(tf.keras.layers.Dense(10,activation="softmax"))
#%% Compiling the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])
model.summary()
#%% Training the model
model.fit(X_train,y_train,epochs=5)
#%% Model Evaluation and Prediction
test_loss,test_accuracy = model.evaluate(X_test,y_test)





