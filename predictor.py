import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# TODO:
# 1) store raw data (complete) note: 52935 features in total
# 2) preprocessing
#       2.1) missing values (complete) note: no na values
#       2.2) scaler
#       2.3) PCA

features = pd.read_csv('train_data/train_expression.csv')
labels = pd.read_csv('train_data/train_labels.csv')

# NOTE:
# check correlation between age and each feature: 
#       47369 features have |corr| > 0.01
#       30240 features have |corr| > 0.05
#       16749 features have |corr| > 0.1
# Idea 1: discard all features with |corr| < threshold and then do PCA
# Idea 2: use PCA to reduce dimension of all features
