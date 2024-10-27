# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:31:44 2023

@author: OKOKPRO
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the AlexNet model architecture
model = Sequential()

# Layer 1
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 2
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 3
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

# Layer 4
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

# Layer 5
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Flatten the output from the convolutional layers
model.add(Flatten())

# Layer 6
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

# Layer 7
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

# Layer 8
model.add(Dense(units=2, activation='softmax'))  # Adjust the number of units based on the number of classes

# Display the model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up the ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('D:/CODE_Lung_cancer/datasetset/train', target_size=(227, 227), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('D:/CODE_Lung_cancer/datasetset/test', target_size=(227, 227), batch_size=32, class_mode='categorical')

# Train the model
r = model.fit(training_set, validation_data=test_set, epochs=10, steps_per_epoch=len(training_set), validation_steps=len(test_set))

# Plot the training and validation loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Save the trained model
model.save('AlexNetModel.h5')
