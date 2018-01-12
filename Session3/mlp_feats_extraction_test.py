import cPickle
import numpy as np
import h5py
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from src.feature_extractors import extract_mlp_features

#Variables
dataset_path = '/imatge/froldan/MIT_split'
IMG_SIZE = 32
BATCH_SIZE = 16
layer2extract = 2
model_fname = './models/mlp_2_hidden_layers.json'
weights_fname = './models/mlp_2_hidden_layers_weights.h5'
train_features_fname = './src/descriptors/train_mlp_2_hidden_layers_'+str(layer2extract)+'.h5'
test_features_fname = './src/descriptors/test_mlp_2_hidden_layers_'+str(layer2extract)+'.h5'
optimizer = 'sgd'

#Obtain model
with open(model_fname, 'r') as f:
    print 'Loading model...'
    model = model_from_json(f.read())
    print 'Model loaded!'
    print 'Compiling baseline model...'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    print 'Model compiled!'

model.load_weights(weights_fname)

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
    image = load_img(filename, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    features.append(extract_mlp_features(model, layer2extract, np.expand_dims(image, axis=0)))
    #print len(features[i][0])
    with h5py.File(train_features_fname, 'w') as f:
        f.create_dataset('image_features', data=np.array(features))
        f.close()
print 'Done!'

#Extract MLP features for each test sample:
features = []
print 'Extracting MLP features for each test sample...'
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    image = load_img(filename, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    features.append(extract_mlp_features(model, layer2extract, np.expand_dims(image, axis=0)))
    #print len(features[i][0])
    with h5py.File(test_features_fname, 'w') as f:
        f.create_dataset('image_features', data=np.array(features))
        f.close()
print 'Done!'