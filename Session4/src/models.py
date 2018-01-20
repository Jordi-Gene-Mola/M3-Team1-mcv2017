from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, concatenate, add, Dropout, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.models import Model
from keras.utils import plot_model


def get_baseline_model(experiment_name, optimizer='adadelta'):
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet')
    plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

    x = base_model.layers[-2].output
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(input=base_model.input, output=x)
    plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_one(experiment_name, optimizer='adadelta'):

    # create the base pre-trained model
    base_model = VGG16(weights='imagenet')
    plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

    x = base_model.get_layer(name='block4_pool').output
    x = AveragePooling2D((2, 2), strides=(2, 2), name='avgpoool')(x)
    x = Flatten()(x)
    #x = Dropout(0.5)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(input=base_model.input, output=x)
    plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_two(experiment_name, optimizer='adadelta'):
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet')
    plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

    x = base_model.get_layer('block4_conv3').output
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten(name='flat')(x)
    x = Dense(2048, activation='relu', name='fc')(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#Task 4 Dropout Layer
def get_model_three(experiment_name, optimizer='adadelta'):

    # create the base pre-trained model
    base_model = VGG16(weights='imagenet')
    plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

    x = base_model.get_layer(name='block4_pool').output
    x = AveragePooling2D((2, 2), strides=(2, 2), name='avgpoool')(x)
    x = Dropout(0.25)(x)     
    x = Flatten()(x)      
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(input=base_model.input, output=x)
    plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
