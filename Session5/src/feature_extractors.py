import cv2
import numpy as np
import h5py
import os
import time


def SIFT_features(SIFTdetector, train_images_filenames, train_labels):
    if not os.path.exists('./src/descriptors/sift_des.npy'):
        print 'Computing SIFT features...'
        init=time.time()
        Train_descriptors = []
        id_des = []
        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            Train_descriptors.append(des)
            id_des.append(i)
        # Transform the descriptors and the labels to numpy arrays
        Train_descriptors_array = np.asarray(Train_descriptors)
        id_des = np.asarray(id_des)

        np.save('./src/descriptors/sift_des', Train_descriptors_array)
        np.save('./src/descriptors/sift_ids', id_des)
        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    else:
        print 'Loading SIFT features...'
        init=time.time()
        Train_descriptors_array = np.load('./src/descriptors/sift_des.npy')
        id_des = np.load('./src/descriptors/sift_ids.npy')
        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    return Train_descriptors_array, id_des

def DenseSIFT_features(SIFTdetector, train_images_filenames):

    if not os.path.exists('./src/descriptors/DenseSift.npy'):
        print 'Computing DenseSIFT features...'
        init=time.time()
        Train_descriptors = []
        id_des = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            #print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kp1=list()
            for x in range(0, gray.shape[0],10):
                for y in range(0, gray.shape[1],10):
                    kp1.append(cv2.KeyPoint(x, y, np.random.randint(10,30)))
            kp1=np.array(kp1)
            kpt, des = SIFTdetector.compute(gray, kp1)
            Train_descriptors.append(des)
            id_des.append(i)

        Train_descriptors_array = np.asarray(Train_descriptors)
        id_des = np.asarray(id_des)

        np.save('./src/descriptors/DenseSift',Train_descriptors_array)
        np.save('./src/descriptors/DenseSift_ids', id_des)

        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    else:
        print 'Loading DenseSift features...'
        init=time.time()
        Train_descriptors_array = np.load('./src/descriptors/DenseSift.npy')
        id_des = np.load('./src/descriptors/DenseSift_ids.npy')

        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    return Train_descriptors_array, id_des

def descriptors_List2Array(descriptors):

    size_descriptors=descriptors[0].shape[1]
    D=np.zeros((np.sum([len(p) for p in descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint+=len(descriptors[i])
    return D

def extract_mlp_features(model, layer, image):
    """Function that takes a Keras model and extracts the features produced by it of the layer specified:
    Args: - Model: Keras model
          - Layer: Integer specifiying the target layer"""

    from keras import backend as K

    get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[layer].output])
    mlp_feats = get_layer_output([image])[0]
    return mlp_feats