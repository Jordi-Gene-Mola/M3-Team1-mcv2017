import cPickle
import numpy as np
import h5py
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def train_knn(descriptors, labels, experiment_filename, save_model=False):
    myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    print 'Fitting KNN...'
    myknn.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving KNN model...'
        cPickle.dump(myknn, open('./models/' + experiment_filename, 'w'))
        print 'KNN model saved!'
    return myknn


def train_random_forest(descriptors, labels, experiment_filename, save_model=False):
    myrf = RandomForestClassifier(n_estimators=10)
    print 'Fitting Random Forest...'
    myrf.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving Random Forest model...'
        cPickle.dump(myrf, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return myrf


def train_gaussian_naive_bayes(descriptors, labels, experiment_filename, save_model=False):
    mygnb = GaussianNB()
    print 'Fitting Gaussian Naive Bayes classifier...'
    mygnb.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving GNB model...'
        cPickle.dump(mygnb, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return mygnb


def train_svm(descriptors, labels, experiment_filename, save_model=False):
    mysvm = svm.SVC(C=5)
    print 'Fitting SVM with RBF kernel...'
    mysvm.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving SVM with RBG kernel model...'
        cPickle.dump(mysvm, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return mysvm


def train_logistic_regression(descriptors, labels, learning_rate=1e-3, L2reg=0.00, num_steps=300000):

    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))

    def softmax(x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
    if not os.path.exists('./src/descriptors/lr_weights.h5'):
        print 'Fitting Logistic Regression Classifier...'
        classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
        target = np.zeros((len(labels), len(classes)))
        #One hot encoding for the different labels:
        for idx, label in enumerate(labels):
            target[idx, classes.index(label)] = 1
        x = descriptors
        y = np.array(target)
        W = np.zeros((descriptors.shape[1], len(classes)),  dtype=np.float64) #shape=(length of features, number of classes)

        for step in range(num_steps):
            p_y_given_x = softmax(np.dot(x, W))
            d_y = y - p_y_given_x
            W += learning_rate * np.dot(x.T, d_y) - learning_rate * L2reg * W
        print 'Done!'
        f = h5py.File('./src/descriptors/lr_weights.h5', 'w')
        f.create_dataset('W', data=W)
        f.close()
    else:
        print 'Loading Logistic regression weights'
        f = h5py.File('./src/descriptors/lr_weights.h5', 'r')
        W = np.array(f['W'])
        f.close()
        print'Finish loading weights'
    return W



