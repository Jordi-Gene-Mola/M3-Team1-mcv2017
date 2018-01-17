from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from src.models import get_baseline_model, get_model_one, get_model_two

train_data_dir = '/imatge/froldan/MIT_split/train'
val_data_dir = '/imatge/froldan/MIT_split/test'
test_data_dir = '/imatge/froldan/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 32
number_of_epoch = 30
experiment_name = 'model_one_test'
WEIGHTS_FNAME = './models/week4/' + experiment_name + '_weights.h5'
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


if model_id == 'baseline':
    model = get_baseline_model(experiment_name)
elif model_id == 1:
    model = get_model_one(experiment_name)
elif model_id ==2:
    model = get_model_two(experiment_name)
else:
    print 'Not a valid model ID'
    quit()

#optimizer=SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=False)
optimizer = Adam(lr=1e-5)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

# preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
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

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')
#Callbacks definition:
checkpoint = ModelCheckpoint(WEIGHTS_FNAME, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
tb = TensorBoard(log_dir='./logs/week4/'+experiment_name+'/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

history = model.fit_generator(train_generator,
                              steps_per_epoch = 1881 // batch_size,
                              epochs=number_of_epoch,
                              validation_data=validation_generator)

result = model.evaluate_generator(test_generator)
print result

