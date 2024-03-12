#Importing the Libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# Construct the absolute path to the CSV file

# Read the CSV file
# dataset = pd.read_csv('/Users/aifertsmacbookpro/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
# dataset = pd.read_csv('Data.csv')
# # Access the "Country" and "Purchased" columns
# x = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1:].values

# Print the "Country" column
# print(x)
# print(y) 


# count = 0
# for i in range(len(x) - 2):
#     array = x[:, i+1]
#     for value in array:
#         if np.isnan(value):
#             count += 1
#     print(f"{count} number of missing data has been identified")

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')
# imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3]) 

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = np.array(ct.fit_transform(x))

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# print(y)

# print(x)
# Importing the necessary libraries
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
dataset = pd.read_csv('titanic.csv')
x = dataset.iloc[:, :].values
y = dataset.iloc[:, 1:2].values

# Identify the categorical data
# let ticket, fare cabin be the categorical data

categorical_features = ['Sex', 'Embarked', 'Pclass']
categorical_indicies = [dataset.columns.get_loc(feature) for feature in categorical_features]
print(categorical_indicies)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_indicies)], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# Implement an instance of the ColumnTransformer class
le = LabelEncoder()
y = le.fit_transform(y)


# Apply the fit_transform method on the instance of ColumnTransformer


# Convert the output into a NumPy array


# Use LabelEncoder to encode binary categorical data


# Print the updated matrix of features and the dependent variable vector

print(x)
print(y)





