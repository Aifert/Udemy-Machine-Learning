# #Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
# # Construct the absolute path to the CSV file

# # Read the CSV file
# dataset = pd.read_csv('/Users/aifertsmacbookpro/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
dataset = pd.read_csv('Data.csv')
# Access the "Country" and "Purchased" columns
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) 

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)



# # Identify the categorical data
# # let ticket, fare cabin be the categorical data

# categorical_features = ['Sex', 'Embarked', 'Pclass']
# categorical_indicies = [dataset.columns.get_loc(feature) for feature in categorical_features]
# print(categorical_indicies)

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_indicies)], remainder="passthrough")
# x = np.array(ct.fit_transform(x))

# # Implement an instance of the ColumnTransformer class
# le = LabelEncoder()
# y = le.fit_transform(y)

# print(x)
# print(y).

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
# print(f"{X_train} X_train")
# print(f"{X_test} X_test")
# print(f"{y_train} y_train")
# print(f"{y_test} y_test")

#Feature Scaling
scaler = StandardScaler()

X_train[:,3:] = scaler.fit_transform(X_train[:,3:])
X_test[:,3:] = scaler.transform(X_test[:,3:])

print(X_train)
print(X_test)