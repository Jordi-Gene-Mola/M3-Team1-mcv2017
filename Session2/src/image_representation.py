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
