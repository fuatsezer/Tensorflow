import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
#%% Import the Data
class_names = ["airplane","automobile","bird","cat","doer","dog","frog","horse","ship","truck"]
(X_train,y_train),(X_test,y_test) =cifar10.load_data()
#%% Image normalization
X_train = X_train / 255.0
X_test = X_test / 255.0
print(X_train.shape)
print(X_test.shape)
plt.imshow(X_test[10])
#%% Defining the model
model = tf.keras.models.Sequential()
#%% Adding the first convonutional layer
#filters : Tamsayı, çıktı uzayının boyutsallığı (yani evrişimdeki çıktı süzgeçlerinin sayısı).
#kernel_size : 2D evrişim penceresinin yüksekliğini ve genişliğini belirten 2 tam sayıdan 
# oluşan bir tamsayı veya tuple / liste. 
#Tüm uzamsal boyutlar için aynı değeri belirtmek için tek bir tam sayı olabilir.
# padding : "valid"veya "same"(büyük / küçük harfe duyarlı değildir). "valid"dolgu yok 
# demektir. "same"Çıktının girişle aynı yükseklik / genişlik boyutuna sahip olacağı şekilde
# girdinin soluna / sağına veya yukarı / aşağıya eşit olarak doldurulmasına neden olur.
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu",
                                 input_shape=[32,32,3]))
#%%  Adding the second convonutional layer and the max pooling layer
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
#pool_size : 2 tam sayılık tamsayı veya demet, maksimumun alınacağı pencere boyutu.
# (2, 2)maksimum değeri 2x2 havuz oluşturma penceresi üzerinden alır.
# Yalnızca bir tam sayı belirtilirse, her iki boyut için aynı pencere uzunluğu kullanılacaktır.
# strides : Tamsayı, 2 tamsayı demeti veya Yok. Adım değerleri. Her bir havuzlama adımı için
# havuzlama penceresinin ne kadar hareket edeceğini belirtir. Yok ise, varsayılan olacaktır
# pool_size.
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid"))
#%% Adding the third convonutional layer
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
#%% Adding the forth convonutional layer and max pooling layer
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid"))
#%% Adding flatten layer
model.add(tf.keras.layers.Flatten()) 
# Adding first fully connected layer
model.add(tf.keras.layers.Dense(128,activation="relu"))
#%% Addding the output layer
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.summary()
#%% Compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer = "Adam",
              metrics=["sparse_categorical_accuracy"])
#%%  Training the model
model.fit(X_train,y_train,epochs=5)
#%% Evaluating the model
test_loss,test_accuracy = model.evaluate(X_test,y_test)





