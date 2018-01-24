from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Input, Conv2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator


train_data_dir = '/imatge/froldan/MIT_split/train'
val_data_dir = '/imatge/froldan/MIT_split/val_set'
test_data_dir = '/imatge/froldan/MIT_split/test_set'
img_width = 224
img_height = 224
batch_size = 32
number_of_epoch = 50
experiment_name = 'my_model_week5_50epochs_lr1e-5'
WEIGHTS_FNAME = './models/week5/' + experiment_name + '_weights.h5'
model_id=2 #model to get


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

optimizer = Adam(lr=1e-5)
#Model definition:
inp = Input(shape=(img_width, img_height, 3, ))
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inp)
x = MaxPooling2D((2, 2), name='maxpooling1')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
x = MaxPooling2D((2, 2), name='maxpooling2')(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
out = Dense(8, activation='softmax', name='predictor')(x)

model = Model(input=inp, output=out)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             zca_whitening=False,
                             rotation_range=0,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')
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
