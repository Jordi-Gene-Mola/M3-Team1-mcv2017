from keras.layers import Input, Dense, concatenate, add, Dropout, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import SGD, Adam

def cnn_model_one(experiment_name):
   img_width = 224
   img_height = 224
   

   model = Sequential()
   model.add(Conv2D(6, (5, 5), activation = 'relu', input_shape=(img_width, img_height, 3),padding='same'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Conv2D(16, (5, 5), activation = 'relu', padding='same'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Conv2D(120, (5, 5), activation = 'relu', padding='same'))
   model.add(Dropout(0.25))

   model.add(Flatten())
   model.add(Dense(512, activation = 'relu'))
   model.add(Dropout(0.5))
   model.add(Dense(8,  activation = 'softmax'))

   plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
   model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
   return model

def cnn_model_two(experiment_name):
   img_width = 224
   img_height = 224
   

   model = Sequential()
   model.add(Conv2D(64, (11, 11), activation = 'relu', input_shape=(img_width, img_height, 3), padding='same', strides=(4,4)))
   model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

   model.add(Conv2D(256, (5, 5), activation = 'relu', padding='same',strides=(1,1)))
   model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

   model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same',strides=(1,1)))
   model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same',strides=(1,1)))
   model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same',strides=(1,1)))
   model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

   model.add(Flatten())
   model.add(Dense(4096, activation = 'relu'))
   model.add(Dropout(0.5))
   model.add(Dense(2048, activation = 'relu'))
   model.add(Dropout(0.5))
   model.add(Dense(128, activation = 'relu'))
   model.add(Dropout(0.5))
   model.add(Dense(8,  activation = 'softmax'))

   plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
   model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
   return model
