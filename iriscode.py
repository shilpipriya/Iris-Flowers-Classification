# Flower classification using NAIVE BAYES

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris_dataset.csv')

#shuffling of dataset
from sklearn.utils import shuffle
dataset=shuffle(dataset)

X=dataset.iloc[:,0:4];
y=dataset.iloc[:,4:5];

#classifying the y-values into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#calculation
count_true,count_false= 0, 0
for i in range(0,38):
    if(y_test[i] == y_pred[i]):
        count_true=count_true+1
    else:
        count_false=count_false+1

accuracy=(count_true)/(count_true+count_false)
print("Accuracy: ", accuracy)
