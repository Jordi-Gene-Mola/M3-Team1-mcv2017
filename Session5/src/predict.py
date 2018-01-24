import numpy as np


def predict_logistic_regression(W, test_set):

    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))

    def softmax(x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

    def one_hot_to_category(encoding):
        classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
        label = np.nonzero(encoding)[0]
        return classes[label[0]]

    # prediction = sigmoid(np.dot(x, W))
    prediction = softmax(np.dot(test_set, W))
    return np.apply_along_axis(one_hot_to_category, axis=1, arr=prediction)
