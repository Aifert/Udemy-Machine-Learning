import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Part 3 - Classification\Section 20 - Random Forest Classification\Python\Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# No feature scaling for trees
# Training the Random Forest Classification model on the training set
from sklearn.ensemble import RandomForest
classifier = RandomForest(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
