import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors.kde
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

mat = np.loadtxt('TP1_train.tsv',delimiter='\t')
matTest = np.loadtxt('TP1_test.tsv',delimiter='\t')

data = shuffle(mat)
dataTest = shuffle(matTest)

X, y = data[:, :-1], data[:, -1] 

Xtest, ytest = matTest[:, :-1], matTest[:, -1]

means = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X- means)/std
Xtest = (Xtest - means)/std

model = GaussianNB()

model.fit(X, y)

y_pred_test = model.predict(Xtest)

accuracy = accuracy_score(ytest, y_pred_test)

print("Error accuracy of Naive Bayes classifier =",1 - accuracy)

print("Accuracy of Naive Bayes classifier =", 100 * accuracy, "%")
