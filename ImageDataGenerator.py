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
print(X_train.shape)
print(X_test.shape)
plt.imshow(X_test[10])
#%% Data Generater
data_gen_train = ImageDataGenerator(rescale=1/255.0)  
data_gen_test = ImageDataGenerator(rescale=1/255.0)  
#%%
train_generator = data_gen_train.flow_from_directory(X_train,target_size=(32,32),batch_size=128,
                                                     class_mode="categorical")