import cv2
import numpy as np
import cPickle
import os
import sys
import time
import Tkinter
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from src.feature_extractors import SIFT_features,  DenseSIFT_features, descriptors_List2Array
from src.image_representation import BoW_hardAssignment, test_BoW_representation
from src.train import train_svm
from src.evaluation import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

start = time.time()

#Variables:
cross_validation = True
extractor = 'SIFT' # SIFT or DenseSIFT
classifier = 'svm' # knn, rf, gnb, svm or lr
kernel_svm='rbf' #Kernel used in svm
n_features=np.array([100,200,300]) #num. of key points detected with SIFT
k=np.array([256,512]) #num. of words
C=1 #Penalty parameter C of the error term in svm algorithm
gamma=0.002 #kernel coefficient for 'rbf', 'poly', and 'sigmoid' in svm algorithm.
bestscore = 0



# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)


NC, DP = np.meshgrid(k,n_features, indexing='ij')
NC_f = NC.flatten()
DP_f = DP.flatten()
for params in xrange(0,len(NC_f)):
    n_feat = DP_f[params]
    clusters = NC_f[params]
    #Feature extraction:
    if extractor=='SIFT':
        myextractor=(cv2.SIFT(n_feat))
        # myextractor=(cv2.xfeatures2d.SIFT_create(nfeatures = n_features))
        Train_descriptors_array = SIFT_features(myextractor, train_images_filenames)
    elif extractor=='DenseSIFT':
        myextractor=(cv2.SIFT(n_feat))
        # myextractor=(cv2.xfeatures2d.SIFT_create(nfeatures = n_features))
        Train_descriptors_array = DenseSIFT_features(myextractor, train_images_filenames)
    else:
        print 'extractor not correct!'

    Train_descriptors=list(Train_descriptors_array)
    D=descriptors_List2Array(Train_descriptors)

    #Getting BoVW with kMeans(Hard Assignment)
    words, visual_words, codebook = BoW_hardAssignment(clusters, D, Train_descriptors)

    # Train an SVM classifier.
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 2e-3, 1e-4],
                         'C': [1, 10, 100]}]


    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)

    print("# Tuning hyper-parameters for")
    print('Using n_features: %s , clusters: %s' % (n_feat, clusters))

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
    clf.fit(D_scaled, train_labels)

    # Calculating test data
    visual_words_test = test_BoW_representation(test_images_filenames, clusters, myextractor, codebook, extractor)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print("Accuracy of: %0.2f" % clf.best_score_)
    if(clf.best_score_>bestscore):
        bestK = clusters
        bestfeat = n_feat
        bestC = clf.best_params_['C']
        bestgamma = clf.best_params_['gamma']
        bestkernel = clf.best_params_['kernel']
    # y_true, y_pred = test_labels, stdSlr.transform(visual_words_test)


#Constants:
experiment_name = extractor + '_' + classifier + '_k' + str(bestK)+ '_C' + str(bestC) + '_gamma' + str(bestgamma)
experiment_filename = experiment_name + '.p'
predictions_filename = './predictions/' + experiment_name + '_predictions.p'

print 'Preparing classifier with best features'
myextractor = (cv2.SIFT(bestfeat))
Train_descriptors = list(Train_descriptors_array)
D = descriptors_List2Array(Train_descriptors)
words, visual_words, codebook = BoW_hardAssignment(bestK, D, Train_descriptors)
clf, stdSlr=train_svm(visual_words, train_labels, experiment_filename, bestkernel, bestC, bestgamma)
print('Best K: %s Best C: %s BestGamma: %s BestFeatures: %s' % (bestK, bestC, bestgamma, bestfeat))


# get all the test data
visual_words_test=test_BoW_representation(test_images_filenames, bestK, myextractor, codebook, extractor)

# Test the classification accuracy
print 'Testing the SVM classifier...'
init=time.time()
accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
end=time.time()
print 'Done in '+str(end-init)+' secs.'
print 'Final accuracy: ' + str(accuracy)

Y_pred=clf.predict(stdSlr.transform(visual_words_test))

cm = plot_confusion_matrix(list(Y_pred), test_labels, experiment_name)

end=time.time()
print 'Everything done in '+str(end-start)+' secs.'
### 69.02% (SIFT)
### 84.51% (DenseSIFT)