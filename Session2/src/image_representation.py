import cv2
import math
import numpy as np
import time
from sklearn import cluster


def BoW_hardAssignment(k, D, ids, spatial_pyramid=True):
    # compute the codebook
    print 'Computing kmeans with ' + str(k) + ' centroids'
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                       reassignment_ratio=10 ** -4, random_state=42)
    codebook.fit(D)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'

    # get train visual word encoding
    print 'Getting Train BoVW representation'
    init = time.time()
    words = codebook.predict(D)
    if spatial_pyramid:
        visual_words = build_pyramid(words, ids, k)
    else:
        visual_words = np.array([np.bincount(words[ids == i], minlength=k) for i in
                            range(0, ids.max() + 1)], dtype=np.float64)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
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

def build_pyramid(prediction, descriptors_indices, k=512):
    levels = [[1, 1], [2, 2], [4, 4]]
    keypoints_shape = map(int, [math.ceil(float(256) / float(6)),
                       math.ceil(float(256) / float(6))])
    kp_i = keypoints_shape[0]
    kp_j = keypoints_shape[1]

    v_words = []

    # Build representation for each image
    for i in range(0, descriptors_indices.max() + 1):

        image_predictions = prediction[descriptors_indices == i]
        #print image_predictions.shape
        #print keypoints_shape
        image_predictions_grid = np.resize(image_predictions, keypoints_shape)

        im_representation = []

        for level in range(0, len(levels)):
            num_rows = levels[level][0]
            num_cols = levels[level][1]
            step_i = int(math.ceil(float(kp_i) / float(num_rows)))
            step_j = int(math.ceil(float(kp_j) / float(num_cols)))

            for i in range(0, kp_i, step_i):
                for j in range(0, kp_j, step_j):
                    hist = np.array(np.bincount(image_predictions_grid[i:i + step_i, j:j + step_j].reshape(-1),
                                                minlength=k))
                    im_representation = np.hstack((im_representation, hist))

        v_words.append(im_representation)

    return np.array(v_words, dtype=np.float64)
