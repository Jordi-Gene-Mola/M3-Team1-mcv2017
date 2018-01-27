import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from src.models_scratch import cnn_model_one, cnn_model_two

train_data_dir = './MIT_split/train'
val_data_dir = './MIT_split/test_set'
test_data_dir = './MIT_split/val_set'
img_width = 224
img_height = 224
batch_size = 32
number_of_epoch = 30
experiment_name = 'test_5'
WEIGHTS_FNAME = './models/' + experiment_name + '_weights_full_net.h5'
model_id='2' #model to get



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

#optimizer=SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=False)
optimizer = Adam(lr=1e-5)
#optimizer = 'adadelta'
if model_id == '1':
    model = cnn_model_one(experiment_name)
elif model_id == '2':
    model = cnn_model_two(experiment_name)
else:
    print 'Not a valid model ID'
    quit()


# preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             preprocessing_function=preprocess_input,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.2,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None)

datagen_val = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             preprocessing_function=preprocess_input,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

validation_generator = datagen_val.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
						   class_mode='categorical')

model.summary()

#Callbacks definition:
checkpoint = ModelCheckpoint(WEIGHTS_FNAME, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
tb = TensorBoard(log_dir='./logs/week5/'+experiment_name+'/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')

history = model.fit_generator(train_generator,
                              steps_per_epoch = 1881 // batch_size,
                              epochs=number_of_epoch,
                              validation_data=validation_generator,
                              callbacks=[checkpoint, tb, reduce_lr, early_stopping])

model.load_weights(WEIGHTS_FNAME)

result = model.evaluate_generator(test_generator)
print result
