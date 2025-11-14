import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
dirname = os.path.dirname(__file__)

tf.__version__

# Importing the dataset
dataset = pd.read_csv(dirname+'\\Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values    # All rows, columns from index 3 to the second last column
y = dataset.iloc[:, -1].values      # All rows, last column
print(x)
print(y)

# Encodes the data in the Gender column to numerical values
# Female = 0, Male = 1
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
print(x)

"""
One Hot Encoding is used to encode the specified column to binary,
1 means the category is present and 0 means it is not.
This is done to improve performance, make the code more compatible with algorithms
and to eliminate ordinality, which can mislead some algorithms to think that one 
category is greater than another and lead to biased predictions.
"""
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()

# It is important to do feature scaling before training the ANN to ensure that all 
# features contribute equally
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_test)
