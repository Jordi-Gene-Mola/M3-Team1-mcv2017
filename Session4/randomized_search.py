from __future__ import print_function

import pickle

import numpy as np
from keras.applications import VGG16
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Flatten, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ParameterSampler

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

#Variables:
train_data_dir = '/imatge/froldan/MIT_split_400/train'
val_data_dir = '/imatge/froldan/MIT_split_400/test'
test_data_dir = '/imatge/froldan/MIT_split_400/test'
img_width = 224
img_height = 224
samples_epoch = 400
val_samples_epoch = 200
test_samples = 200
epochs_fc = 10
epochs_whole = 10
num_experiments = 5

hyperparameters = {
    'batch_size': np.array([8, 64]),
    'learning_rate':  np.logspace(-6, -3, 10 ** 4),
}

results = {}

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             zca_whitening=False,
                             rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.,
                             zoom_range=0.2,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=preprocess_input)


randomized_hyperparams = list(ParameterSampler(hyperparameters, n_iter=num_experiments))
for ind, params in enumerate(randomized_hyperparams):
    # Create model
    base_model = VGG16(weights='imagenet', input_shape=(img_width, img_height, 3))
    x = base_model.get_layer(name='block4_pool').output
    x = AveragePooling2D((2, 2), strides=(2, 2), name='avgpoool')(x)
    x = Flatten()(x)
    # x = Dropout(0.5)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    print('Experiment {} of {}'.format(ind + 1, num_experiments))

    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    print('batch size: {}'.format(batch_size))
    print('learning rate: {}'.format(learning_rate))


    experiment_name = 'batchsize_{}_adam_lr_{:.4G}'.format(
        batch_size,
        learning_rate)

    optimizer = Adam(lr=learning_rate)
    train_generator = datagen.flow_from_directory(train_data_dir,
                                                  shuffle=True,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    validation_generator = datagen.flow_from_directory(val_data_dir,
                                                       shuffle=True,
                                                       target_size=(img_width, img_height),
                                                       batch_size=batch_size,
                                                       class_mode='categorical')

    print('Training model with VGG weights frozen...')
    for layer in base_model.layers:
        layer.trainable = False
    #Callbacks definition:
    checkpoint = ModelCheckpoint('/imatge/froldan/mcv-m3-team1/models/cnn_optimization_fc_{}.h5'.format(experiment_name), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    tb = TensorBoard(log_dir='./logs/week4/'+experiment_name+'_fc/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history_fc = model.fit_generator(train_generator,
                                     steps_per_epoch=400 // batch_size,
                                     epochs=epochs_fc,
                                     validation_data=validation_generator,
                                     callbacks=[checkpoint, tb, reduce_lr])


    print('Training the whole network...')
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #Callbacks definition:
    checkpoint = ModelCheckpoint('/imatge/froldan/mcv-m3-team1/models/cnn_optimization_full_{}.h5'.format(experiment_name), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    tb = TensorBoard(log_dir='./logs/week4/'+experiment_name+'_full/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    history_full = model.fit_generator(train_generator,
                                       steps_per_epoch=400 // batch_size,
                                       epochs=epochs_whole,
                                       validation_data=validation_generator,
                                       callbacks=[checkpoint, tb, reduce_lr])


    print('Evaluating validation set...')
    result = model.evaluate_generator(validation_generator)
    print('Loss: {:.2f} ; Accuracy: {:.2f} %'.format(result[0], result[1]))
    results.update({
        experiment_name: {
            'accuracy': result[1]*100,
            'loss': result[0]
        }
    })
    with open('./results/cnn_optimization.p', 'wb') as f:
        pickle.dump(results, f)

