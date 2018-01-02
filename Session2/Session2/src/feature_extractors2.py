import cv2
import numpy as np
import h5py
import os
import cPickle

def SIFT_features(SIFTdetector, train_images_filenames, train_labels):

    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        #print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])

    return Train_descriptors, Train_label_per_descriptor

def SURF_features(SURFdetector, train_images_filenames, train_labels):

    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SURFdetector.detectAndCompute(gray, None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print str(len(kpt)) + ' extracted keypoints and descriptors'

    return Train_descriptors, Train_label_per_descriptor

def descriptors_List2Array(descriptors):

    size_descriptors=descriptors[0].shape[1]
    D=np.zeros((np.sum([len(p) for p in descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint+=len(descriptors[i])
    return D
