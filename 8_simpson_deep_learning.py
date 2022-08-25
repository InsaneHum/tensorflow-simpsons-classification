import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

IMG_SIZE = (80, 80)
channels = 1
char_path = r'simpsons_dataset/simpsons_dataset'

# sort characters by size of database
char_dict = {}

for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)

# grab 10 characters with the most images
characters = []

for i in char_dict:
    characters.append(i[0])
    if len(characters) == 10:
        break

'''
# save data for the first time
# create training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# cv.imshow('First pic', train[0][0])
# cv.waitKey(0)

# separate the training set into the feature set and labels and reshapes the data into a 4 dimensional tensor to feed into tensorflow

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

print('Separation done, saving files...')
np.save('features.npy', featureSet)
np.save('labels.npy', labels)
print('Done saving files')
'''

# load data
featureSet = np.load('features.npy')
labels = np.load('labels.npy')

# normalize the featureSet ==> (0, 1)
featureSet = caer.normalize(featureSet)

# convert labels from numerical integers to binary class vectors
labels = to_categorical(labels, len(characters))

# split the featureSet and labels into training sets and validation sets with a validation ration of 20% (20% of the data goes into the validation set, 80% for the training set)
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.1)

# convert x_train, x_val, y_train, y_val to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

# remove variables to save memory
del featureSet
del labels
gc.collect()

# image data generator (image generator that synthesizes new images from existing images to add randomness to the network (makes it perform better))
BATCH_SIZE = 2
EPOCHS = 20

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# creating the model
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), loss='binary_crossentropy', decay=1E-6, learning_rate=0.001, momentum=0.9, nesterov=True)

# print the summary of the model
model.summary()

# create callbacks list (learning rate schedule that will schedule the learning rate at specific intervals so that the network can be trained better)
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

# train the model
training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), validation_steps=len(y_val)//BATCH_SIZE, callbacks=callbacks_list)


model.save('simpsons_deep_learning_model')
