import cv2
import cPickle
import math
import numpy as np
import time
import os
import math
import scipy.cluster.vq as vq
from sklearn import cluster
from yael.yael import ynumpy

def BoW_hardAssignment(k, D, Train_descriptors):
    #compute the codebook
    print 'Computing kmeans with '+str(k)+' centroids'
    init=time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
    codebook.fit(D)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'

    # get train visual word encoding
    print 'Getting Train BoVW representation'
    init=time.time()
    visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        words=codebook.predict(Train_descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return words, visual_words, codebook



def test_BoW_representation(test_images_filenames, k, myextractor, codebook, extractor):

    print 'Getting Test BoVW representation'
    init=time.time()

    if extractor=='SIFT':
        visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
        for i in range(len(test_images_filenames)):
            filename=test_images_filenames[i]
            #print 'Reading image '+filename
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            kpt,des=myextractor.detectAndCompute(gray,None)
            words=codebook.predict(des)
            visual_words_test[i,:]=np.bincount(words,minlength=k)

    elif extractor=='DenseSIFT':
        visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
        for i in range(len(test_images_filenames)):
            filename=test_images_filenames[i]
            #print 'Reading image '+filename
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            kp1=list()
            for x in range(0, gray.shape[0],10):
                for y in range(0, gray.shape[1],10):
                    kp1.append(cv2.KeyPoint(x, y, np.random.randint(10,30)))
            kp1=np.array(kp1)
            kpt, des = myextractor.compute(gray, kp1)
            words=codebook.predict(des)
            visual_words_test[i,:]=np.bincount(words,minlength=k)
    else:
        print 'extractor not correct!'

    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return visual_words_test


def build_spatial_pyramid(descriptor, level):

    assert 0 <= level <= 2, "Level Error"
    step_size = 4
    h = 256 / step_size
    w = 256 / step_size
    idx_crop = np.resize(np.array(range(len(descriptor))), [h, w])
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
            idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

def spatial_pyramid_matching(descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(descriptor, 0)
        words = [obtain_word_hist(crop, codebook) for crop in pyramid]
        return np.asarray(words).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(descriptor, 0)
        pyramid += build_spatial_pyramid(descriptor, 1)
        words = [obtain_word_hist(crop, codebook) for crop in pyramid]
        words_level_0 = 0.5 * np.asarray(words[0]).flatten()
        words_level_1 = 0.5 * np.asarray(words[1:]).flatten()
        return np.concatenate((words_level_0, words_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(descriptor, 0)
        pyramid += build_spatial_pyramid(descriptor, 1)
        pyramid += build_spatial_pyramid(descriptor, 2)
        words = [obtain_word_hist(crop, codebook) for crop in pyramid]
        words_level_0 = 0.25 * np.asarray(words[0]).flatten()
        words_level_1 = 0.25 * np.asarray(words[1:5]).flatten()
        words_level_2 = 0.5 * np.asarray(words[5:]).flatten()
        return np.concatenate((words_level_0, words_level_1, words_level_2))

def obtain_word_hist(feature, codebook):
   words, _ = vq.vq(feature, codebook)
   word_hist, bin_edges = np.histogram(codebook, bins=range(codebook.shape[0] + 1), normed=True)
   return word_hist

def fisher_vectors(descriptors, idxs, k, image_filenames=None, myextractor=None):
    if descriptors is None:
        descriptors = []
        id_des = []
        for i in range(len(image_filenames)):
            filename = image_filenames[i]
            #print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = myextractor.detectAndCompute(gray, None)
            descriptors.append(des)
            id_des.append(i)

        # Transform the descriptors and the labels to numpy arrays
        descriptors_matrix = descriptors[0]
        ids_matrix = np.array([id_des[0]] * descriptors[0].shape[0])
        for i in range(1, len(descriptors)):
            descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
            ids_matrix = np.hstack((ids_matrix, np.array([id_des[i]] * descriptors[i].shape[0])))
        descriptors = descriptors_matrix
        idxs = ids_matrix

    """pca = PCA(n_components=20)
    pca.fit(descriptors)
    descriptors_matrix = pca.transform(descriptors)
    descriptors = np.float32(descriptors_matrix)"""

    # train GMM
    gmm = ynumpy.gmm_learn(np.float32(descriptors), k)
    image_fvs = np.array([ynumpy.fisher(gmm, descriptors[idxs == i],
                                        include=['mu', 'sigma']) for i in range(0, idxs.max() + 1)])

    # make one matrix with all FVs
    image_fvs = np.vstack(image_fvs)

    # normalizations are done on all descriptors at once

    # power-normalization
    image_fvs = np.sign(image_fvs) * np.abs(image_fvs) ** 0.5

    # L2 normalize
    norms = np.sqrt(np.sum(image_fvs ** 2, 1))
    image_fvs /= norms.reshape(-1, 1)
    return image_fvs
