import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training regressor model using rbf kernel
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Predicting result and changing the result from scaled to unscaled
scaled_result = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

print(scaled_result)


plt.scatter((sc_x.inverse_transform(x)) , (sc_y.inverse_transform(y)), color = "red")
plt.plot((sc_x.inverse_transform(x)), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)) , color = "blue")
plt.title("Truth of Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.show()


unscaled_x = sc_x.inverse_transform(x)
x_grid = np.arange(min(unscaled_x), max(unscaled_x), 0.1)
x_grid = x_grid.reshape((len(x_grid)),1)
plt.scatter((sc_x.inverse_transform(x)) , (sc_y.inverse_transform(y)), color = "red")
plt.plot((sc_x.inverse_transform(x)), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)) , color = "blue")
plt.title("Truth of Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.show()