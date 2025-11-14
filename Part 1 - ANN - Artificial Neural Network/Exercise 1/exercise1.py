import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
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

'''
Making a single prediction (for example the customer with the following features)
Geography: France, 
Credit Score: 600, 
Gender: Male (1), 
Age: 40, 
Tenure: 3, 
Balance: 60000, 
Number of Products: 2, 
Has Credit Card: Yes (1), 
Is Active Member: Yes (1), 
Estimated Salary: 50000
The result is also scaled using the same scaler as used for the training set
'''
output = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(output)
print("Stays in the bank" if output > 0.5 else "Leaves the bank")

# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

'''
Making the Confusion Matrix:
A confusion matrix is a table that is used in classification 
problems to assess where errors in the model were made.
They can be created by predictions made from the model compared 
to the actual results from the data.
'''
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracy)