import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors.kde
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

mat = np.loadtxt('TP1_train.tsv',delimiter='\t')
matTest = np.loadtxt('TP1_test.tsv',delimiter='\t')

plt.figure(1, figsize=(12, 8))

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
score = model.score(Xtest, ytest)
scoreOr = model.score(X, y)

print("Error accuracy of Gaussian Naive Bayes classifier for training =", 1 - scoreOr)
print("Error accuracy of Gaussian Naive Bayes classifier for testing =", 1 - score)

skf = StratifiedKFold(n_splits=5)

allR = []
allV = []

def calc_fold(X, Y, train_ix, valid_ix, gv):
    classifier = SVC(C=1, kernel= 'rbf', gamma = gv)
    classifier.fit(X[train_ix], Y[train_ix])
    tr_err = 1 - accuracy_score(Y[train_ix], classifier.predict(X[train_ix]))
    va_err = 1 - accuracy_score(Y[valid_ix], classifier.predict(X[valid_ix]))
    return tr_err, va_err

best_gamma = 0.0
best_value = 1
for gv in np.arange(0.2, 6.2, 0.2):
    tr_err = va_err = 0
    for tr_ix, va_ix, in skf.split(y, y):
        r, v = calc_fold(X, y, tr_ix, va_ix, gv)
        tr_err += r
        va_err += v
    if(r < best_value):
        best_gamma = gv
        best_value = r
            
    print(gv, tr_err/5, va_err/5)
    allR.append(tr_err/5)
    allV.append(va_err/5)
    
    

classifier = SVC(C=1, kernel= 'rbf', gamma = best_gamma)
classifier.fit(X, y)
print("Error: ", 1 - accuracy_score(ytest, classifier.predict(Xtest)), best_gamma)
plt.plot(np.arange(0.2, 6.2, 0.2), allR, label='Training Error')
plt.plot(np.arange(0.2, 6.2, 0.2), allV, label='Validation Error')
plt.grid('true')
plt.yticks(np.arange(0.01, 0.05, 0.01))
plt.legend(['Training Error', 'Validation Error'], loc=7)
plt.savefig('tp1.png')
plt.show()
