import cv2
import numpy as np
import h5py
import os

def SIFT_features(SIFTdetector, train_images_filenames, train_labels):

    if not os.path.exists('./src/descriptors/sift.h5'):
        # read the just 30 train images per class
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Train_label_per_descriptor = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i]) < 30:
                print 'Reading image ' + filename
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt, des = SIFTdetector[0].detectAndCompute(gray, None)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(train_labels[i])
                print str(len(kpt)) + ' extracted keypoints and descriptors'

        # Transform everything to numpy arrays
        D = Train_descriptors[0]
        L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

        for i in range(1, len(Train_descriptors)):
            D = np.vstack((D, Train_descriptors[i]))
            L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
        f = h5py.File('./src/descriptors/sift.h5', 'w')
        f.create_dataset('D', data=D)
        f.create_dataset('L', data=L)
        f.close()
    else:
        sift_descriptor = h5py.File('./src/descriptors/sift.h5', 'r')
        D = np.array(sift_descriptor['D'])
        L = np.array(sift_descriptor['L'])
        sift_descriptor.close()
    return D, L


def n_SIFT_features(SIFTdetectors, train_images_filenames, train_labels):
    if not os.path.exists('./src/descriptors/n_sift.h5'):
        # read the just 30 train images per class
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Train_label_per_descriptor = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i]) < 30:
                print 'Reading image ' + filename
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                for SIFTdetector in SIFTdetectors:
                    kpt, des = SIFTdetector.detectAndCompute(gray, None)
                    Train_descriptors.append(des)
                    Train_label_per_descriptor.append(train_labels[i])
                print str(len(kpt)) + ' extracted keypoints and descriptors'

        # Transform everything to numpy arrays

        D = Train_descriptors[0]
        L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

        for i in range(1, len(Train_descriptors)):
            D = np.vstack((D, Train_descriptors[i]))
            L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
        f = h5py.File('./src/descriptors/n_sift.h5', 'w')
        f.create_dataset('D', data=D)
        f.create_dataset('L', data=L)
        f.close()
    else:
        n_sift_descriptor = h5py.File('./src/descriptors/n_sift.h5', 'r')
        D = np.array(n_sift_descriptor['D'])
        L = np.array(n_sift_descriptor['L'])
        n_sift_descriptor.close()
    return D, L


def SURF_features(SURFdetector, train_images_filenames, train_labels):

    if not os.path.exists('./src/descriptors/surf.h5'):
        # read the just 30 train images per class
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Train_label_per_descriptor = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i]) < 30:
                print 'Reading image ' + filename
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt, des = SURFdetector[0].detectAndCompute(gray, None)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(train_labels[i])
                print str(len(kpt)) + ' extracted keypoints and descriptors'

        # Transform everything to numpy arrays

        D = Train_descriptors[0]
        L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

        for i in range(1, len(Train_descriptors)):
            D = np.vstack((D, Train_descriptors[i]))
            L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
        f = h5py.File('./src/descriptors/surf.h5', 'w')
        f.create_dataset('D', data=D)
        f.create_dataset('L', data=L)
        f.close()
    else:
        sift_descriptor = h5py.File('./src/descriptors/surf.h5', 'r')
        D = np.array(sift_descriptor['D'])
        L = np.array(sift_descriptor['L'])
        sift_descriptor.close()
    return D, L