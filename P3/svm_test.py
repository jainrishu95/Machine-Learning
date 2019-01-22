import numpy as np

def objective_function(X, y, w, lamb):
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    y = y.reshape((y.shape[0], 1))
    minus = np.subtract(1, np.multiply(y, np.dot(X, w)))
    max_ = np.sum(np.maximum(0, minus)) * 1/X.shape[0]
    obj_value = (lamb / 2 * np.dot(w.T, w)) + max_
    return obj_value

X = [[1,3,4],[2,5,4],[6,3,2]]
y = [0,1,0]
w = [0.2,0.4,-0.2]
lamb = 0.01
print(objective_function(X, y, w, lamb))