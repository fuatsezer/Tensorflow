import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
#%% Loading the pre-trained model (MobileNetV2)
IMG_SHAPE = (32,32,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,
                                               weights="imagenet")
base_model.summary()
#%%
base_model.trainable = False
#%% 
print(base_model.output)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(base_model)
prediction_layer = tf.keras.layers.Dense(1,activation="sigmoid")(global_average_layer)