# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:33:37 2023

@author: OKOKPRO
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the GoogleNet/Inception-v1 architecture
def inception_module(x, filters):
    branch1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    branch3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3)

    branch5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5)

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    output = Concatenate(axis=-1)([branch1, branch3, branch5, branch_pool])
    return output

input = Input(shape=(224, 224, 3))
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, [64, 96, 128, 16, 32, 32])
x = inception_module(x, [128, 128, 192, 32, 96, 64])
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, [192, 96, 208, 16, 48, 64])
x = inception_module(x, [160, 112, 224, 24, 64, 64])
x = inception_module(x, [128, 128, 256, 24, 64, 64])
x = inception_module(x, [112, 144, 288, 32, 64, 64])
x = inception_module(x, [256, 160, 320, 32, 128, 128])
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, [256, 160, 320, 32, 128, 128])
x = inception_module(x, [384, 192, 384, 48, 128, 128])
x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
output = Dense(2, activation='softmax')(x)

model = Model(input, output)

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up the ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('D:/CODE_Lung_cancer/datasetset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('D:/CODE_Lung_cancer/datasetset/test', target_size=(224, 224), batch_size=32, class_mode='categorical')

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
model.save('GoogleNet.h5')
