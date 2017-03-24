import csv
import numpy as np
import cv2
import pdb
from sklearn.utils import shuffle
import sklearn

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


def generator(samples, batch_size = 32):
    correction = 0.20
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            print("\nRunning batch " + str(offset))
            batch_samples = samples[offset:(offset + batch_size)]

            images = []
            angles = []

            for batch_sample in batch_samples:

                center_image = get_image(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                center_image = get_image(batch_sample[0])
                center_image = cv2.flip(center_image, 1)
                center_angle *= -1
                images.append(center_image)
                angles.append(center_angle)

                left_image = get_image(batch_sample[1])
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)

                left_image = cv2.flip(left_image, 1)
                left_angle *= -1
                images.append(left_image)
                angles.append(left_angle)

                right_image = get_image(batch_sample[2])
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)

                right_image = cv2.flip(right_image, 1)
                right_angle *= -1
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.layers import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) # normalizes all images
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = "relu"))
model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = "relu"))
model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = "relu"))
model.add(Conv2D(64, (3, 3), strides = (2, 2), activation = "relu"))
model.add(Conv2D(64, (3, 3), strides = (2, 2), activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), epochs = 3, verbose = 1)
model.save('model.h5')
