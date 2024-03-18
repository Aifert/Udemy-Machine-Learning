import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Part 3 - Classification/Section 17 - Kernel SVM/Python/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
#Splitting into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# training the model using rbf svm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#predicting the test set results
y_pred = classifier.predict(x_test)

y_strict = classifier.predict(sc.transform([[30,87000]]))
print(y_strict)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))