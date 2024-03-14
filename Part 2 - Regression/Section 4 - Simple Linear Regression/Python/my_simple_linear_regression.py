import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the data
dataset = pd.read_csv('Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')

#Seperating the features and result
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#Seperating into training and data in a 80 / 20 split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=5)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Visualising the training results
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

## Visualising the test results
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()