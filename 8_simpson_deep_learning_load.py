import os
import caer
import numpy as np
import cv2 as cv
from tensorflow import keras

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
print(characters)

model = keras.models.load_model('simpsons_deep_learning_model')


IMG_SIZE = (80, 80)

img = cv.imread('simpsons_dataset/kaggle_simpson_testset/kaggle_simpson_testset/homer_simpson_40.jpg')
# cv.imshow('Bart Simpson?', img)


def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image


img = prepare(img)

predictions = model.predict(img)

print(characters[np.argmax(predictions[0])])
cv.waitKey(0)


