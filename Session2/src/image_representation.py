import cv2
import numpy as np
import time
from sklearn import cluster


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

def extract_pyramid_bins(levels, kpt, des, dimensions):

    keypoints, descriptors = [], []
    if levels == []:
        return keypoints, descriptors
    x_divisions, y_divisions = levels[0]
    min_limit_x, min_limit_y = dimensions[0], dimensions[1]
    max_limit_x, max_limit_y = dimensions[2], dimensions[3]

    x_step = (max_limit_x-min_limit_x) / float(x_divisions)
    y_step = (max_limit_y-min_limit_y) / float(y_divisions)

    for x_div in range(x_divisions):
        for y_div in range(y_divisions):

            bin_kpt = []
            bin_des = []

            min_x,max_x = min_limit_x + x_step*x_div, min_limit_x + x_step*(x_div+1)
            min_y,max_y = min_limit_y + y_step*y_div, min_limit_y + y_step*(y_div+1)

            for i, kp in enumerate(kpt):
                if (kp.pt[0] >= min_x and kp.pt[0] < max_x) and (kp.pt[1] >= min_y and kp.pt[1] < max_y):
                    bin_kpt.append(kpt[i])
                    bin_des.append(des[i])

            keypoints.append(bin_kpt)
            descriptors.append(bin_des)

            level_dimensions = [min_x, min_y, max_x, max_y]
            lower_kps, lower_des = extract_pyramid_bins(levels[1:], bin_kpt, bin_des, level_dimensions)

            keypoints += lower_kps
            descriptors += lower_des
    return keypoints, descriptors
