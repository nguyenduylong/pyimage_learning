from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from LeNetModel import LeNet
from ReadH5File import ReadH5Data
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
# initialize the data and labels
print("[INFO] loading images...")

data, labels = ReadH5Data.readData(args['dataset'])
# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)

# for imagePath in imagePaths:
#     image = cv2.imread(imagePath)
#     image = cv2.resize(image, (28, 28))
#     image = img_to_array(image)
#     data.append(image)
#     label = imagePath.split(os.path.sep)[-2]
#     #todo : add labels data

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state = 42)
# todo : num_classes and Y data
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = LeNet.build(width=28, height=28, depth=3, classes=10)
opt = Adam(lr=INIT_LR, decay= INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer= opt , metrics=['accuracy'])

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

model.save(args['model'])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])