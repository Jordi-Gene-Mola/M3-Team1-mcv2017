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
from src.image_representation import BoW_hardAssignment, test_BoW_representation, spatial_pyramid_matching
from src.train import train_svm
from src.evaluation import plot_confusion_matrix, rcurve

start = time.time()

#Variables:
extractor = 'SIFT' # SIFT or DenseSIFT
classifier = 'svm' # knn, rf, gnb, svm or lr
kernel_svm='rbf' #Kernel used in svm ('rbf' or 'precomputed')
n_features=300 #num. of key points detected with SIFT
k=512 #num. of words
C=1 #Penalty parameter C of the error term in svm algorithm
gamma=0.002 #kernel coefficient for 'rbf', 'poly', and 'sigmoid' in svm algorithm.
spatial_pyramid = True

#Constants:
experiment_name = extractor + '_' + classifier + '_k' + str(k)+ '_C' + str(C) + '_gamma' + str(gamma)
experiment_filename = experiment_name + '.p'
predictions_filename = './predictions/' + experiment_name + '_predictions.p'

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)


#Feature extraction:
if extractor=='SIFT':
    #myextractor=(cv2.SIFT(nfeatures=300))
    myextractor=(cv2.xfeatures2d.SIFT_create(nfeatures = n_features))
    Train_descriptors_array, labels_matrix, ids_matrix = SIFT_features(myextractor, train_images_filenames, spatial_pyramid)
elif extractor=='DenseSIFT':
    #myextractor=(cv2.SIFT(nfeatures=300))
    myextractor=(cv2.xfeatures2d.SIFT_create(nfeatures = n_features))
    Train_descriptors_array = DenseSIFT_features(myextractor, train_images_filenames)
else:
    print 'extractor not correct!'
Train_descriptors=list(Train_descriptors_array)
#D=descriptors_List2Array(Train_descriptors)
D = Train_descriptors_array.astype(np.uint32)

#Getting BoVW with kMeans(Hard Assignment)
words, visual_words, codebook = BoW_hardAssignment(k, D, ids_matrix)

if spatial_pyramid:
    print 'Creating Spatial Pyramid...'
    visual_words = [spatial_pyramid_matching(D[i], words, 1, ids_matrix, k) for i in xrange(len(D))]
    print 'Done!'
# Train an SVM classifier.
clf, stdSlr=train_svm(visual_words, train_labels, experiment_filename, kernel_svm, C, gamma)


# get all the test data 
visual_words_test=test_BoW_representation(test_images_filenames, k, myextractor, codebook, extractor)

# Test the classification accuracy
print 'Testing the SVM classifier...'
init=time.time()
accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
end=time.time()
print 'Done in '+str(end-init)+' secs.'
print 'Final accuracy: ' + str(accuracy)

testDescriptors=stdSlr.transform(visual_words_test)
Y_pred=clf.predict(testDescriptors)
cm = plot_confusion_matrix(list(Y_pred), test_labels, experiment_name)
ROCcurve=rcurve(testDescriptors, test_labels,clf)

end=time.time()
print 'Everything done in '+str(end-start)+' secs.'
### 69.02% (SIFT)
### 84.51% (DenseSIFT)

