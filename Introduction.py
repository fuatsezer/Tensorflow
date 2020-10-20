import tensorflow as tf
import numpy as np
print(tf.__version__)
#%% constant
tensor_20 = tf.constant([[23,4],[32,51]])
print(tensor_20)
print(tensor_20.shape)
#%%
print(tensor_20.numpy())
numpy_tensor = np.array([[23,4],[32,51]])
tensor_to_numpy = tf.constant(numpy_tensor)
print(tensor_to_numpy)
#%% Variable 
tf2_variable = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
print(tf2_variable)
print(tf2_variable.numpy())
print(tf2_variable[0,2].assign(100))
print(tf2_variable*2)
#%% operation
tensor = tf.constant([[1,2],[3,4]])
print(tensor)
print(tensor + 2)
print(tensor * 5)
print(tensor ** 4)
print(np.sqrt(tensor))
tensor2 = tf.constant([[4,2],2,4])
print(np.dot[tensor,tensor2])
#%% String 
tf_string = tf.constant("Tensorflow")
print(tf_string)
print(tf.strings.length(tf_string))
tf_string_array = tf.constant(["Tensorflow","AI","Deep Learning"])
for string in tf_string_array:
    print(string)