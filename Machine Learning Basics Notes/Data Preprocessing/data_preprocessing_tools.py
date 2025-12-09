# Data Pre-Processing Tools

# Import libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

dirname = os.path.dirname(__file__) + "\\example_datasets\\"

# Import the dataset
dataset = pd.read_csv(dirname+"Data.csv")

# Independent Variables
X = dataset.iloc[:, :-1].values
# Dependent Variables
Y = dataset.iloc[:, -1].values

# Using SimpleImputer to find the mean of the columns to replace missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data to numerical data because machine learning models require numerical input

# Encoding independent variables
# OneHotEncoding converts categorical data into binary columns, ColumnTransformer applies the changes to the specified columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding dependent variables
# LabelEncoder converts categorical labels into numerical labels
le = LabelEncoder()
Y = le.fit_transform(Y)

# Splitting the dataset into random training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Feature Scaling, transforming variables to a common scale
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)