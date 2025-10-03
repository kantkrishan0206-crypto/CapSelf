from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten, Conv2D,
                   MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import optimizers
from time import strftime, localtime
import warnings
import os
import pickle
from DataGenerator import DataGenerator
import numpy as np
import h5py
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model



def base_Alexnet(tileSize=64, numPuzzles=9):
    """
    Returns Alexnet with stride of 4 in the first conv layre
    """
    inputShape = (tileSize, tileSize, 3)
    inputTensor = Input(inputShape)
    x=Conv2D(filters=96, kernel_size=(11,11),strides=(2,2), padding='same')(inputTensor) # the only difference here is we use stride of 2
    x=Activation('relu')(x)
    # Pooling 
    x=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    # Batch Normalisation before passing it to the next layer
    x=BatchNormalization()(x)

    # 2nd Convolutional Layer
    x=Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same')(x)
    x=Activation('relu')(x)
    # Pooling
    x=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    # Batch Normalisation
    x=BatchNormalization()(x)

    # 3rd Convolutional Layer
    x=Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x=Activation('relu')(x)
    # Batch Normalisation
    x=BatchNormalization()(x)

    # 4th Convolutional Layer
    x=Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x=Activation('relu')(x)
    # Batch Normalisation
    x=BatchNormalization()(x)

    # 5th Convolutional Layer
    x=Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x=Activation('relu')(x)
    # Pooling
    x=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    # Batch Normalisation
    x=BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    
    model = Model(inputTensor, x, name='base_Alexnet')
    return model

def Alex_net(tileSize=64, numPuzzles=9, hammingSetSize=100):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (tileSize, tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = base_Alexnet()
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)  # Reconsider what axis to merge
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


def Resnet_50(tileSize=64, numPuzzles=9, hammingSetSize=100):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (tileSize, tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = ResNet50(include_top=False, weights=None,
        pooling="avg")
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)  
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


USE_MULTIPROCESSING = False

if USE_MULTIPROCESSING:
    n_workers = 8
    warnings.warn('Generators are not thread safe!', UserWarning)
else:
    n_workers = 1


hdf5_path = ' ' #path to HDF5 dataset
batch_size = 32
num_epochs = 50
hamming_set_size = 100 #number of permutations
#choose either alex-net or resnet model
model = Resnet_50()
#model = Alex_net()
model.summary()

# Open up the dataset
hdf5_file = h5py.File(hdf5_path,'r')
train_dataset = hdf5_file['train_img']
val_dataset = hdf5_file['val_img']
test_dataset = hdf5_file['test_img']

max_hamming_set = np.loadtxt(" ") #path to permutations text file
dataGenerator = DataGenerator(batchSize=batch_size,maxHammingSet=max_hamming_set[:hamming_set_size])

# Output all data from a training session into a dated folder
outputPath = './model_data/{}'.format(strftime('%b_%d_%H:%M:%S', localtime()))
os.makedirs(outputPath)
checkpointer = ModelCheckpoint(
    outputPath +
    '/weights_improvement.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True)
reduce_lr_plateau = ReduceLROnPlateau(
    monitor='val_loss', patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
# opt=optimizers.SGD(
#     learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(dataGenerator.generate(train_dataset),
                              epochs=num_epochs,
                              steps_per_epoch=train_dataset.shape[0] // batch_size,
                              validation_data=dataGenerator.generate(
                                  val_dataset),
                              validation_steps=val_dataset.shape[0] // batch_size,
                              use_multiprocessing=USE_MULTIPROCESSING,
                              workers=n_workers,
                              shuffle=False,
                              callbacks=[checkpointer, reduce_lr_plateau, early_stop])

scores = model.evaluate_generator(
    dataGenerator.generate(test_dataset),
    steps=test_dataset.shape[0] //
    batch_size,
    workers=n_workers,
    use_multiprocessing=USE_MULTIPROCESSING)

# Output the test loss and accuracy
print("Test loss: {}".format(scores[0]))
print("Test accuracy: {}".format(scores[1]))

# Save the train and val accuracy history to file
with open(outputPath + '/history.pkl', 'wb') as history_file:
    pickle.dump(history.history, history_file)
# Save the test score accuracy
with open(outputPath + '/test_accuracy.pkl', 'wb') as test_file:
    pickle.dump(scores, test_file)

model.save(outputPath + '/model.hdf5')
