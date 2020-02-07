import numpy as np
from collections import Counter

def euclidian(x, y):
  length = len(x)
  total = 0
  for i in range(length):
    total += (x[i] - y[i]) ** 2
  return total ** 0.5


class KNeighbors:
  def __init__(self, k):
    self.k = k
    self.X = None
    self.y = None
  
  def fit(self, X, y):
    self.X = X
    self.y = np.array(y)
  
  def predict(self, test):
    distances = []
    for datapoint in self.X:
      distances.append(euclidian(datapoint, test))
    classes = self.y[np.argsort(distances)[:self.k]]
    classes = Counter(classes)
    print(classes)
    return classes.most_common(1)[0][0]

X = [[1,2,3],
     [1,2,3],
     [4,5,5],
     [4,3,4]]

y = [0,0,1,1]

kneighbors = KNeighbors(1)
kneighbors.fit(X,y)
print(kneighbors.predict([1,1,3]))
