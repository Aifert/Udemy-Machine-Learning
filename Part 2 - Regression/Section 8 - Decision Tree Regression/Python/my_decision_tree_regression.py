import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
X_grid = np.c_[x_grid]
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
