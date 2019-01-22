import numpy as np
import json
import linear_regression as lr
import knn as knn


def weights_to_string(ws, is_int=False):
    ws = np.array(ws)
    ws[np.abs(ws) < 1e-10] = 0
    ws = ws.flatten().tolist()
    s = ''
    for w in ws:
        s = s + ('%.3f,' if not is_int else '%d,') % w
    return s


def test_wrapper(fn, n_lines, *argv):
    try:
        res = fn(*argv)
    except Exception as e:
        res = ['ERROR @' + fn.__name__ + '\t' + str(e), ] * n_lines
    return res


"""
Functions to test regression
"""


def test_linear_regression_noreg(Xtrain, ytrain):
    result = []
    w = lr.linear_regression_noreg(Xtrain, ytrain)

    result.append('[TEST LinearRegressionNonReg]' + str(len(w)) + ",")
    result.append('[TEST LinearRegressionNonReg]' + weights_to_string(w))
    return result


def test_regularized_linear_regression(Xtrain, ytrain, lambd):
    result = []
    w = lr.regularized_linear_regression(Xtrain, ytrain, lambd)

    result.append('[TEST RegularizedLinearRegression]' + str(len(w)) + ",")
    result.append('[TEST RegularizedLinearRegression]' + weights_to_string(w))
    return result


def test_tune_lambda(Xtrain, ytrain, Xval, yval, lambds):
    result = []
    bestlambd = lr.tune_lambda(Xtrain, ytrain, Xval, yval, lambds)

    result.append(('[TEST TuneLambda]%.3f,') % bestlambd)
    return result


def test_test_error(Xtrain, ytrain, Xval, yval, lambds):
    result = []
    bestlambd = lr.tune_lambda(Xtrain, ytrain, Xval, yval, lambds)
    wbest = lr.regularized_linear_regression(Xtrain, ytrain, bestlambd)
    mse = lr.test_error(wbest, Xtest, ytest)

    result.append(('[TEST TestError]%.3f,') % mse)
    return result


"""
Functions to test knn
"""


def test_compute_distances(KNN_Xtrain, KNN_Xval):
    result = []
    dists = knn.compute_distances(KNN_Xtrain, KNN_Xval)
    result.append('[TEST KNNComputeDistances]' + weights_to_string(dists))
    return result


def test_predict_labels(KNN_Xtrain, KNN_Xval, KNN_ytrain):
    result = []
    dists = knn.compute_distances(KNN_Xtrain, KNN_Xval)
    ypred = knn.predict_labels(5, KNN_ytrain, dists)
    result.append('[TEST KNNPredictLabels]' + weights_to_string(ypred, True))
    return result


def test_compute_accuracy(KNN_Xtrain, KNN_Xval, KNN_ytrain, KNN_yval):
    result = []
    dists = knn.compute_distances(KNN_Xtrain, KNN_Xval)
    ypred = knn.predict_labels(5, KNN_ytrain, dists)
    acc = knn.compute_accuracy(KNN_yval, ypred)
    result.append(('[TEST KNNComputeAccuracy]%.3f,') % acc)
    return result


def test_find_best_k(KNN_Xtrain, KNN_Xval, K, KNN_ytrain, KNN_yval):
    result = []
    dists = knn.compute_distances(KNN_Xtrain, KNN_Xval)
    best_k, validation_accuracy = knn.find_best_k(K, KNN_ytrain, dists, KNN_yval)

    result.append(('[TEST KNNBestK]%d') % best_k + ',')
    result.append('[TEST KNNAccuracyList]' + weights_to_string(validation_accuracy))
    return result


if __name__ == '__main__':
    result = []
    # Running results for Regression
    Xtrain, ytrain, Xval, yval, Xtest, ytest = lr.data_processing()
    lambd = 5.0
    lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]

    result += test_wrapper(test_linear_regression_noreg, 2, Xtrain, ytrain)
    result += test_wrapper(test_regularized_linear_regression, 2, Xtrain, ytrain, lambd)
    result += test_wrapper(test_tune_lambda, 1, Xtrain, ytrain, Xval, yval, lambds)
    result += test_wrapper(test_test_error, 1, Xtrain, ytrain, Xval, yval, lambds)

    # # Running tests for KNN
    input_file = 'mnist_subset.json'
    with open(input_file) as json_data:
        data = json.load(json_data)
    KNN_Xtrain, KNN_ytrain, KNN_Xval, KNN_yval, KNN_Xtest, KNN_ytest = knn.data_processing(data)
    K = [1, 3, 5, 7, 9]

    result += test_wrapper(test_compute_distances, 1, KNN_Xtrain, KNN_Xval)
    result += test_wrapper(test_predict_labels, 1, KNN_Xtrain, KNN_Xval, KNN_ytrain)
    result += test_wrapper(test_compute_accuracy, 1, KNN_Xtrain, KNN_Xval, KNN_ytrain, KNN_yval)
    result += test_wrapper(test_find_best_k, 1, KNN_Xtrain, KNN_Xval, K, KNN_ytrain, KNN_yval)
    with open('output_hw1.csv', 'w') as f:
        for line in result:
            f.write(line[:-1] + '\n')