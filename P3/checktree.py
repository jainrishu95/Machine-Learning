from sklearn.metrics import accuracy_score
import decision_tree

# load data
X_train = [[0, 0, 0, 0], [0, 0, 0, 1],[1, 0, 0, 0],[2, 1, 0, 0],[2, 2, 1, 0],[2, 2, 1, 1],[1, 2, 1, 1],[0, 1, 0, 0],[0, 2, 1, 0],
           [2, 1, 1, 0],[0, 1, 1, 1],[1, 1, 0, 1],[1, 0, 1, 0],[2, 1, 0, 1]]

y_train = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]

# set classifier
dTree = decision_tree.DecisionTree()

# training
dTree.train(X_train, y_train)
y_est_train = dTree.predict(X_train)
train_accu = accuracy_score(y_est_train, y_train)
print('train_accu', train_accu)

# # testing
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)