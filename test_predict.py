from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import argparse
import pickle
import cv2
import os
import imutils

pokemons_names = [ 'Charmander', 'Pikachu', 'Bulbasaur', 'Squirtle', 'Caterpie', 'Butterfree', 'Mankey', 'Machop', 'Cubone', 'Koffing', 'Mr. Mime', 'Snorlax']

model = load_model('./pokemon/model/pokemon_model.model')
# label_pickle = pickle.loads(open('./pokemon/model/labels.pickle', 'rb').read())
for dirname, _, filenames in os.walk('./pokemon/test_data/'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        image = cv2.imread(filepath)
        output = image.copy()
        image = cv2.resize(image, (96, 96))
        image = image.astype('float') / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        # label_id = label_pickle.classes_[idx]
        label = pokemons_names[idx]
        label = "{}: {:.2f}%".format(label, proba[idx] * 100)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Output", output)
        cv2.waitKey(0)