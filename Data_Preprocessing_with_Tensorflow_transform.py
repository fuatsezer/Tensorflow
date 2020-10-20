from __future__ import  print_function
import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema
#%% Data Preprocessing
data = pd.read_csv("pollution_small.csv")
print(data.head(10))
#%% Dropping the data column
features = data.drop("Date",axis=1)
print(features.head())
#%%
dict_features = list(features.to_dict("index").values())
print(dict_features[:2])
#Defining the dataset metadata
data_meta = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        "no2":tf.FixedLenFeature([],tf.float32),
        "so2":tf.FixedLenFeature([],tf.float32),
        "pm10":tf.FixedLenFeature([],tf.float32),
        "soot":tf.FixedLenFeature([],tf.float32)}
        )
    )
print(data_meta)