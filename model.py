import csv
import numpy as np
import cv2
import pdb
from sklearn.utils import shuffle
import sklearn

# Path for training data input
base_path = "/home/carnd/data/"
lines = []

# load all lines from driving log into the lines list.
# note - remove the 1st row having the labels
with open(base_path + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Function that takes the path from the driving log and returns corresponding image
def get_image(path):
    source_path = path
    filename = source_path.split('/')[-1]
    current_path = base_path + "IMG/" + filename
    image = cv2.imread(current_path)
    return image

# generator to augment and train the model with images in batches of 32
def generator(samples, batch_size = 32):
    correction = 0.20
    num_samples = len(samples)
    print("\n Number of sample " + str(num_samples))

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


# NOTE - Commented out code is retained to switch between
# older and newer versions of Keras.
# My local machine has newer version. But Udacity's starter kit has
# older keras. The older one is installed on AWS.
# Newer keras versions using Conv2D over Convolution2D
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
# from keras.layers import Conv2D
from keras.layers import Convolution2D
from keras.layers import Dropout

model = Sequential()
# Normalize input images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
# Crop top 50 pixel and bottom 20 pixel blocks of images
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = "relu"))
# model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = "relu"))
# model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = "relu"))

# Convolution layer - output 24, kernel size 5 x 5, stride of 2x2
# Uses RELU activation to introduce non-linearity
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = "relu"))

# model.add(Conv2D(64, (3, 3), strides = (2, 2), activation = "relu"))
# model.add(Conv2D(64, (3, 3), strides = (2, 2), activation = "relu"))

model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation = "relu"))

model.add(Flatten())

# Fully connected layer with output of 100
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

from sklearn.model_selection import train_test_split

# Fetch training and validation samples
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# Call generator on batch of training images
train_generator = generator(train_samples, batch_size = 32)
# Call generator on batch of validation images
validation_generator = generator(validation_samples, batch_size = 32)

# Use mean squared error for loss function and Adam Optimizer
model.compile(loss='mse', optimizer='adam')
# model.fit_generator(train_generator, steps_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), epochs = 3, verbose = 1)

# Start training the model, uses 3 Epochs since in our tests, beyond 3 epochs, losses started increasing
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch = 3, verbose = 1)
# Save model
model.save('model.h5')# Start training the model, uses 3 Epochs since in our tests, beyond 3 epochs, losses started increasing
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
