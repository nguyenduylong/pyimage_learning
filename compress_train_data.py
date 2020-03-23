import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import random
import cv2
import os
import h5py

IMAGE_DIMS = (96, 96, 3)

data = []
labels = []
dataset_path = './pokemon'
imagePaths = sorted(list(paths.list_images(dataset_path)))
random.seed(42)
random.shuffle(imagePaths)
pokemons_names = [ 'Charmander', 'Pikachu', 'Bulbasaur', 'Squirtle', 'Caterpie', 'Butterfree', 'Mankey', 'Machop', 'Cubone', 'Koffing', 'Mr. Mime', 'Snorlax']

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(pokemons_names.index(label))

data =  np.array(data, dtype="float")
labels = np.array(labels, dtype="int8")
print(labels)
with h5py.File('./pokemon/compress_data/pokemon_data.h5', 'w') as f:
    f.create_dataset('images', data=data)
    f.create_dataset('labels', data=labels)


