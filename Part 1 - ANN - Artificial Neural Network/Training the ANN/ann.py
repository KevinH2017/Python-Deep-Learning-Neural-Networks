import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential

tf.__version__

dataset = pd.read_csv('./Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values    # All rows, columns from index 3 to the second last column
y = dataset.iloc[:, -1].values      # All rows, last column

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initializing the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(Dense(units=1, activation='sigmoid'))

# Compiling the ANN with these parameters
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to a training set
ann.fit(x_train, y_train, batch_size=32, epochs=100)

