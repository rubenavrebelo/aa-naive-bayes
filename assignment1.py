import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity
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

skf = StratifiedKFold(n_splits=5)

"""""""""""""""""""""""""""""""""""""""""""""
Naive Bayes (KDE)
"""""""""""""""""""""""""""""""""""""""""""""

def cross_valid_kde(X, Y, train_ix, valid_ix, bw):
    Prob, Kdes = naiveBayes(X[train_ix], Y[train_ix], bw)
    tr_err = 1 - accuracy_score(Y[train_ix], predict(X[train_ix], Prob, Kdes))
    va_err = 1 - accuracy_score(Y[valid_ix], predict(X[valid_ix], Prob, Kdes))
    return tr_err, va_err

def predict(X, prob, Kdes):
    predict_class = np.zeros(len(X))
    predictionsZero = np.ones(len(X)) * np.log(prob[0])
    predictionsOne = np.ones(len(X)) * np.log(prob[1])
    
    for i in range(X.shape[1]):
        predictionsZero += Kdes[i][0].score_samples(X[:, [i]])
        predictionsOne += Kdes[i][1].score_samples(X[:, [i]])
        
    for i in range(len(X)):
        if predictionsOne[i] > predictionsZero[i]:
            predict_class[i] = 1
    return predict_class
        
    
def fit(X0, X1, numFeatures, bw):
    Kdes = []
    for feature in range(numFeatures):
        kde0 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X0[:, [feature]])
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X1[:, [feature]])
        Kdes.append((kde0, kde1))
    return Kdes

def attributeClasses(X, Y):
    x0 = X[Y == 0, :]
    x1 = X[Y == 1, :]
    return x0, x1

def naiveBayes(X, Y, bw):
    featureZeros, featureOnes = attributeClasses(X, Y) 
    prob = (len(featureZeros[0]) / len(X), len(featureOnes[1]) / len(X))
    Kdes = fit(featureZeros, featureOnes, 4, bw)
    return prob, Kdes

best_value = 1
best_bw = 10
    
allRKDE = []
allVKDE = []

for bw in np.arange(0.02, 0.6, 0.02):
        tr_err = va_err = 0
        for tr_ix, va_ix, in skf.split(X, y):
            r, v = cross_valid_kde(X, y, tr_ix, va_ix, bw)
            tr_err += r
            va_err += v
        if(v < best_value):
            best_value = v
            best_bw = bw        
        allRKDE.append(tr_err/5)
        allVKDE.append(va_err/5)
        
def aproximateNormalTest(X, N):
    omega = np.sqrt((N*(X/N) * (1 - X/N)))
    interval = X + 1.96 * omega
    return N * X/N, interval


prob, Kdes = naiveBayes(X, y, best_bw)
ypredkde = predict(Xtest, prob, Kdes)
accuracy = accuracy_score(ytest, ypredkde)
print("Naive Bayes Kde Error ", 1 -accuracy)
Np0, interval = aproximateNormalTest(accuracy * len(ytest), len(ytest))
print("Naive Bayes Normal Test interval:", Np0, " +- ", interval)

plt.plot(np.arange(0.02, 0.6, 0.02), allRKDE, label='Training Error')
plt.plot(np.arange(0.02, 0.6, 0.02), allVKDE, label='Validation Error')
plt.grid('true')
plt.yticks(np.arange(0.03, 0.08, 0.01))
plt.legend(['Training Error', 'Validation Error'], loc=7)
plt.savefig('NB.png')
plt.show()
plt.close()
        


"""""""""""""""""""""""""""""""""""""""""""""
GaussianNB
"""""""""""""""""""""""""""""""""""""""""""""
model = GaussianNB()
model.fit(X, y)
ypred_gaussian = model.predict(Xtest)
score = model.score(Xtest, ytest)
scoreOr = model.score(X, y)

print("Error accuracy of Gaussian Naive Bayes classifier for training =", 1 - scoreOr)
print("Error accuracy of Gaussian Naive Bayes classifier for testing =", 1 - score)
Np0, interval = aproximateNormalTest((1-accuracy) * len(ytest), len(ytest))
print("Gaussian Naive Bayes Normal Test interval:", Np0, " +- ", interval)


allR = []
allV = []

"""""""""""""""""""""""""""""""""""""""""""""
SVM
"""""""""""""""""""""""""""""""""""""""""""""

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
    for tr_ix, va_ix, in skf.split(X, y):
        r, v = calc_fold(X, y, tr_ix, va_ix, gv)
        tr_err += r
        va_err += v
    if(v < best_value):
        best_gamma = gv
        best_value = v
            
    allR.append(tr_err/5)
    allV.append(va_err/5)
    
    
classifier = SVC(C=1, kernel= 'rbf', gamma = best_gamma)
classifier.fit(X, y)
ysvmpredict = classifier.predict(Xtest)
accuracySVM = accuracy_score(ytest, ysvmpredict)

plt.figure(1, figsize=(12, 8))
plt.plot(np.arange(0.2, 6.2, 0.2), allR, label='Training Error')
plt.plot(np.arange(0.2, 6.2, 0.2), allV, label='Validation Error')
plt.grid('true')
plt.yticks(np.arange(0.01, 0.06, 0.01))
plt.legend(['Training Error', 'Validation Error'], loc=7)
plt.savefig('SVM.png')
plt.show()
plt.close() 

"""""""""""""""""""""""""""""""""""""""""""""
Mcnemar
"""""""""""""""""""""""""""""""""""""""""""""

def mcnemarTest(e01, e10):
    calc = (abs(e01 -e10)** 2) / (e01 + e10)
    return calc

def e01e10(classifier1, classifier2):
    e01 = 0
    e10 = 0
    for i in range(len(ytest)):
        if classifier1[i] != ytest[i] and classifier2[i] == ytest[i]:
            e01 +=1
        elif classifier1[i] == ytest[i] and classifier2[i] != ytest[i]:
            e10 +=1
    return e01, e10
        

print("Error for SVM Classifier: ", 1 - accuracySVM, best_gamma)
Np0, interval = aproximateNormalTest( (1- accuracySVM) * len(ytest), len(ytest))
print("SVM Normal Test interval:", Np0, " +- ", interval)

e01, e10 = e01e10(ypredkde,ypred_gaussian)
print ("Mcnemar for KDE Naive Bayes & Gaussian NB",
       mcnemarTest(e01, e10))
e01, e10 = e01e10(ypredkde,ysvmpredict)
print ("Mcnemar for KDE Naive Bayes & SVM",
       mcnemarTest(e01, e10))
e01, e10 = e01e10(ypred_gaussian,ysvmpredict)
print ("Mcnemar for Gaussian NB & SVM",
       mcnemarTest(e01, e10))
