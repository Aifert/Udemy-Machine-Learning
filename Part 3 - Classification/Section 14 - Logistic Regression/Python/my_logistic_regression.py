import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split   
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting a new result with Logistic Regression
y_pred = classifier.predict([x_test[0]])

y_array_pred = classifier.predict(x_test)

y_array_pred = np.reshape(y_array_pred, (len(y_array_pred), 1))

np.set_printoptions(precision=2)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_array_pred)
accuracy = accuracy_score(y_test, y_array_pred)

print(cm)
print(accuracy)