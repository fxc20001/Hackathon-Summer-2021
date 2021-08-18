import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# TODO:
# 1) store raw data (complete) note: 52935 features in total
# 2) preprocessing
#       2.1) missing values (complete) note: no na values
#       2.2) scaler
#       2.3) PCA

# read data
features = pd.read_csv('train_data/train_expression.csv')
labels = pd.read_csv('train_data/train_labels.csv')

# number of features of raw data
size_features = len(features.columns)

# get features with correlation > 0.05 with label
corr = [labels['age'].corr(features[features.columns[i]]) for i in range(size_features)]
sorted_corr = [i for i in sorted(enumerate(corr), key=lambda x:abs(x[1]), reverse=True)]

features_strong_index = [i[0] for i in sorted_corr if abs(i[1]) > 0.05]
features_strong = features[[features.columns[i] for i in features_strong_index]]


# NOTE:
# check correlation between age and each feature: 
#       47369 features have |corr| > 0.01
#       30240 features have |corr| > 0.05
#       16749 features have |corr| > 0.1
# Idea 1: discard all features with |corr| < threshold and then do PCA
# Idea 2: use PCA to reduce dimension of all features
