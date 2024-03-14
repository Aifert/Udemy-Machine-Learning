import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
x = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print("Predicted Results, Actual Results")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
