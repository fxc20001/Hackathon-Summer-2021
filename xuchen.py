import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read data
df = pd.read_csv('train_data/train_expression.csv')
label = pd.read_csv('train_data/train_labels.csv')

size = len(df.columns)

corr = [label['age'].corr(df[df.columns[i]]) for i in range(size)]
sorted_corr = [i for i in sorted(enumerate(corr), key=lambda x:abs(x[1]), reverse=True)]




