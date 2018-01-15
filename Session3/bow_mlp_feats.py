import cPickle
import h5py
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from src.feature_extractors import extract_mlp_features, descriptors_List2Array
from src.image_representation import BoW_hardAssignment
from src.train import train_svm
from utils import *
#user defined variables
PATCH_SIZE  = 64
BATCH_SIZE  = 16
layer2extract = 1
DATASET_DIR = '../MIT_split'
PATCHES_DIR = '../MIT_split_patches'
experiment_name = 'patch_based_mlp_2hidden'
WEIGHTS_FNAME = './models/' + experiment_name + '_weights.h5'
model_fname = './models/' + experiment_name + '.json'
train_features_fname = './src/descriptors/train_patch_based_mlp_2hidden_'+str(layer2extract)+'.h5'
test_features_fname = './src/descriptors/test_patch_based_mlp_2hidden_'+str(layer2extract)+'.h5'
optimizer = 'sgd'
kernel = 'rbf'
C = 1 #Penalty parameter C of the error term in svm algorithm
gamma = 0.002 #kernel coefficient for 'rbf', 'poly', and 'sigmoid' in svm algorithm.

#Obtain model
with open(model_fname, 'r') as f:
    print 'Loading model...'
    model = model_from_json(f.read())
    print 'Model loaded!'
    print 'Compiling baseline model...'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    print 'Model compiled!'

model.load_weights(WEIGHTS_FNAME)

# read the train and test files
train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat','r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat','r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat','r'))

#Extract MLP features for each train sample:
features = []
print 'Extracting MLP features for each train sample...'
for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    im = Image.open(filename)
    patches = create_patches(im, PATCH_SIZE)
    patches = [img_to_array(Image.fromarray(patch)) / 255. for patch in patches]
    dense_feats = []
    for patch in patches:
        dense_feats.append(extract_mlp_features(model, layer2extract, np.expand_dims(patch, axis=0)))
    features.append(np.array(dense_feats))
    #print len(features[i][0])
    #with h5py.File(train_features_fname, 'w') as f:
    #    f.create_dataset('image_features', data=np.array(features))
    #    f.close()
    #print 'Done!'

descriptors = list(np.asarray(features))
D = descriptors_List2Array(descriptors)
print 'Creating codebook...'
words, visual_words, codebook = BoW_hardAssignment(32, D, np.asarray(features))
print 'Finished!'
clf, stdSlr = train_svm(visual_words, train_labels, experiment_name, kernel, C, gamma)

directory = DATASET_DIR+'/test'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0
visual_words_test = np.zeros((len(test_images_filenames), 32), dtype=np.float32)
for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory, class_dir)):
      im = Image.open(os.path.join(directory, class_dir, imname))
      patches = create_patches(im, PATCH_SIZE)
      patches = [img_to_array(Image.fromarray(patch))/255. for patch in patches]
      dense_feats = []
      for patch in patches:
          dense_feats.append(extract_mlp_features(model, layer2extract, np.expand_dims(patch, axis=0)))
      words = codebook.predict(np.array(dense_feats))
      visual_words_test[i, :] = np.bincount(words, minlength=32)

# Test the classification accuracy
print 'Testing the SVM classifier...'
accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
print 'Final accuracy: ' + str(accuracy)
