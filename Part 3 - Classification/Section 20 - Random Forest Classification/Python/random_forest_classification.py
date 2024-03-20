import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data
dataset = pd.read_csv('Part 3 - Classification/Section 20 - Random Forest Classification/Python/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# splitting into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# training the random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier.fit(x_train, y_train)

#prediciting the results
y_pred = classifier.predict(x_test)

#displaying results
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)
print(accuracy_score(y_pred, y_test))