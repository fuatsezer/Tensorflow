from __future__ import  print_function
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdw
#%% Simple data analysis
data = pd.read_csv("pollution_small.csv")
training_data = data[:1600]
print(training_data.describe())
#%%
test_set = data[1600:]
#%% Generate training data statistic
train_stats = tfdw.generate_statistics_from_dataframe(data)
schema = tfdw.infer_schema(statistics=train_stats)
tfdw.display_schema(schema)
#%%
test_stats = tfdw.generate_statistics_from_dataframe(test_set)
#%% Anomalies detection
anomalies = tfdw.validate_statistics(statistics=test_stats,schema=schema)
print(tfdw.display_anomalies(anomalies))
#%% 
test_set_copy = test_set.copy()
test_set_copy.drop("soot",axis=1,inplace=True)
test_set_copy_stats = tfdw.generate_statistics_from_dataframe(test_set_copy)
#%%
anomalies_new = tfdw.validate_statistics(statistics=test_set_copy_stats,schema=schema)
print(tfdw.display_anomalies(anomalies_new))
