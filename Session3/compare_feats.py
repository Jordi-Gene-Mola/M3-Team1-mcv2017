import cPickle
import h5py
import numpy as np

from src.train import train_svm


#Variables:
train_features_fname = './src/descriptors/train_mlp_2_hidden_layers_1.h5'
test_features_fname = './src/descriptors/test_mlp_2_hidden_layers_1.h5'
kernel = 'rbf'
C = 1 #Penalty parameter C of the error term in svm algorithm
gamma = 0.002 #kernel coefficient for 'rbf', 'poly', and 'sigmoid' in svm algorithm.

#Load train and test MLP image features:
train_image_data = h5py.File(train_features_fname, 'r')['image_features']
train_image_data = np.reshape(train_image_data, [train_image_data.shape[0], train_image_data.shape[2]])
test_image_data = h5py.File(test_features_fname, 'r')['image_features']
test_image_data = np.reshape(test_image_data, [test_image_data.shape[0], test_image_data.shape[2]])

train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))

#Train SVM classifier using MLP extracted features:
clf, stdSlr = train_svm(train_image_data, train_labels, 'mlp_feats_svm', kernel, C, gamma)

accuracy = 100*clf.score(stdSlr.transform(test_image_data), test_labels)
print 'Final accuracy: ' + str(accuracy)
#End-to-end network obtained aprox 63%, SVM using MLP feats obtained 65% --> mlp_2_hidden_layers_2
#                                       SVM using MLP feats obtained 53.28% --> mlp_2_hidden_layer_1



