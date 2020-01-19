"""
Simple OLS

(X.T * X)^-1 * (X.T * Y)

X - inoput, Y - output
Add another column with all one's to the X matrix to calculate the intercept
"""

from numpy.linalg import inv
np.set_printoptions(suppress=True)

X = np.array([[1,1],[2,1],[3,1],[4,1],[5,1]])
Y = np.array([10,20,30,40,50])
np.dot(inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y))
