import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# plt.scatter(x,y, color="red")
# plt.plot(x, lin_regressor.predict(x), color='blue') # this one prints salary from a linear regression
# plt.title("Truth or Bluff (Linear Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# plt.scatter(x,y, color="red")
# plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue') # this one prints salary from a linear regression
# plt.title("Truth or Bluff (Polynomial Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

print(lin_regressor.predict([[6.5]]))

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))