"""
How to proceed with this file:
Make sure to have an account on Udemy since the dataset is available there
1. Download zip file dataset folder and unzip it. Copy paste this link to download the file:  https://www.udemy.com/course/machinelearning/learn/lecture/19453374#notes
2. Download this file in the same folder along with the dataset
3. Can change the epoch value. Increase if you want to increase the number of total rounds it should go through the dataset, but be careful as it takes time and it may result
in overfitting. I have set it to 5 only
4. Make sure to have tensorflow version 2.2.0, if not follow:
    4.1. pip install tensorflow
    4.2. pip install keras
"""
# Convolutional Neural Network

### Importing the libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

## Part 1 - Data Preprocessing

### Preprocessing the Training set

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

### Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

## Part 2 - Building the CNN

### Initialising the CNN

cnn = tf.keras.models.Sequential()

### Step 1 - Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=[64, 64, 3]))

### Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

### Step 4 - Full Connection

# Adding the input layer and the first hidden layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Step 5 - Output Layer

# Add the output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Use activation='softmax' if using other than binary classification
# Since we are using activation at last layer for binary classification
# We use activation='sigmoid'

## Part 3 - Training the CNN

### Compiling the CNN

cnn.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy'])

### Training the CNN on the Training set and evaluating it on the Test set

cnn.fit(x=training_set, validation_data=test_set, epochs=5)

## Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)
