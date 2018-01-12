from utils import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/imatge/froldan/MIT_split'
experiment_name = 'mlp_2_hidden_layers'
MODEL_FNAME = './models/' + experiment_name + '.h5'
WEIGHTS_FNAME = './models/' + experiment_name + '_weights.h5'

if not os.path.exists(DATASET_DIR):
  colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


colorprint(Color.BLUE, 'Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='input'))
model.add(Dense(units=2048, activation='relu',name='fc1'))
model.add(Dense(units=1024, activation='relu', name='fc2'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
print('Saving baseline model...')
model_json = model.to_json()
with open('./models/'+experiment_name+'.json', 'w') as f:
    f.write(model_json)
print('Baseline model saved')

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

colorprint(Color.BLUE, 'Done!\n')

if os.path.exists(MODEL_FNAME):
  colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

colorprint(Color.BLUE, 'Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

#Callbacks definitions:
checkpoint = ModelCheckpoint(WEIGHTS_FNAME, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')
tb = TensorBoard(log_dir='./logs/'+experiment_name+'/', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False,
            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, tb])

colorprint(Color.BLUE, 'Done!\n')
colorprint(Color.BLUE, 'Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
colorprint(Color.BLUE, 'Done!\n')