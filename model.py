import csv
import numpy as np
import cv2
import pdb

base_path = "/Users/Akshay/Desktop/p3/data/"
lines = []

with open(base_path + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def get_image(path):
    source_path = path
    filename = source_path.split('/')[-1]
    current_path = base_path + "IMG/" + filename
    image = cv2.imread(current_path)
    return image

images = []
measurements = []
correction = 0.20

for line in lines[1:]:
    image = get_image(line[0])
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    image = image.copy()
    image = cv2.flip(image, 1)
    measurement *= -1
    images.append(image)
    measurements.append(measurement)

    left_image = get_image(line[1])
    images.append(left_image)
    measurement = float(line[3]) + correction
    measurements.append(measurement)

    right_image = get_image(line[2])
    images.append(right_image)
    measurement = float(line[3]) - correction
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) # normalizes all images
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, subsample = (2, 2), activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs = 3, verbose = 1)
model.save('model.h5')
