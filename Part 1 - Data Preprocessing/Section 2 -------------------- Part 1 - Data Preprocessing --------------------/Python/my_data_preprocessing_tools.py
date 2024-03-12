#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Construct the absolute path to the CSV file

# Read the CSV file
# dataset = pd.read_csv('/Users/aifertsmacbookpro/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
dataset = pd.read_csv('Data.csv')
# Access the "Country" and "Purchased" columns
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# Print the "Country" column
print(x)
print(y) 


# count = 0
# for i in range(len(x) - 2):
#     array = x[:, i+1]
#     for value in array:
#         if np.isnan(value):
#             count += 1
#     print(f"{count} number of missing data has been identified")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder




print(x)

