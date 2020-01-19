"""
Basic Implementation of Naive Bayes Algorithm without Laplace Smoothing

THEORY :

Bayes Rule ->

P(A/B) = (P(B/A) * P(A))/P(B)

In context of ML

P(y|x1,x2,…,xn) = (P(x1,x2,…,xn|y) * P(y))/P(x1,x2,…,xn)
P(y|x1,x2,…,xn) = (P(x1|y) * P(x2|y) * P(x3|y) * ... * P(Xn|y) * P(y))/P(x1) * P(x2) * P(x3) * ... * P(Xn)

Since the denominator P(x1) * P(x2) * P(x3) * ... * P(Xn) will be the same for all the classes we do not consider it
and just use the above equation using proportionality

P(y|x1,x2,…,xn) ~ P(x1|y) * P(x2|y) * P(x3|y) * ... * P(Xn|y) * P(y)

So basically we calculate the P(y|x1,x2,…,xn) for all the classes along with the test features and one with the highest 
probability is the output y

"""

import numpy as np

data = np.array([[0,0,1,1],
    [0,1,1,1],
    [0,1,1,1],
    [1,1,0,1],
    [0,1,0,1],
    [0,1,1,1],
    [1,0,0,0],
    [1,1,0,0],
    [1,0,1,0],
    [1,0,0,0]])

X = pd.DataFrame({0: data[:, 0], 1: data[:, 1], 2: data[:, 2], 3: data[:, 3]})
test = [1,1,0]
classes_ = X[3].unique()
class_probs = []

for class_idx in range(len(classes_)):
    class_count = len(X[X[3] == classes_[class_idx]])
    internal_prob = len(X[X[3] == classes_[class_idx]]/len(X))
    for idx in range(len(X.columns[:-1])):
        feature_count = len(X[(X[idx] == test[idx]) & (X[3] == classes_[class_idx])])
        internal_prob = internal_prob * (feature_count/class_count)
    class_probs.append(internal_prob)

print(classes_[np.argmax(class_probs)])
