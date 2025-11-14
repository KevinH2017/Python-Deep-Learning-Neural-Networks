import tensorflow as tf
import numpy as np
from keras.layers import Rescaling, RandomFlip, RandomZoom, RandomRotation, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import image_dataset_from_directory, load_img, img_to_array

tf.__version__

'''
Load dataset from directory:

labels are inferred from the directory structure
label_mode is set to 'binary' for binary classification
batch_size is set to 32 to process 32 images at a time
image_size is set to (64, 64) to resize images to 64x64 pixels
shuffle is set to True to shuffle the data
'''
train_image_gen = image_dataset_from_directory(
    directory="Part 2 - CNN - Convolutional Neural Network\\example_datasets\\training_set",
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    image_size=(64, 64),
    shuffle=True)
print("Training dataset imported successfully!")

# class_names is a dictionary containing maps from class names (strings) to corresponding integers
class_names = train_image_gen.class_names

test_image_gen = image_dataset_from_directory(
    directory="Part 2 - CNN - Convolutional Neural Network\\example_datasets\\test_set",
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    image_size=(64, 64),
    shuffle=False)      # Keeps test data in order for evaluation
print("Test dataset imported successfully!")

'''
Define preprocessing and augmentation pipeline:

Recaling is used to scale pixel values to [0, 1]
RandomFlip is used to flip images horizontally
RandomZoom is used to zoom into images randomly
RandomRotation is used to rotate images randomly
'''
cnn = tf.keras.Sequential([
    Rescaling(1./255),
    RandomFlip('horizontal'),
    RandomZoom(0.2),
    RandomRotation(0.2)
])

# Apply preprocessing
# lambda function is used to apply data_augmentation to each batch of images
training_set = train_image_gen.map(lambda x, y: (cnn(x), y))
print("Data augmentation applied successfully!")

'''
Add 2D convolution layer:

filters is set to 32 to learn 32 different features
kernel_size is set to (3, 3) to use a 3x3 filter size
activation is set to 'relu' to introduce non-linearity
input_shape is set to (64, 64, 3) to match the input image dimensions
'''
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

'''
Add MaxPooling2D layer:

pool_size is set to 2 to reduce the spatial dimensions by half
strides is set to 2 to move the pooling window by 2 pixels
'''
cnn.add(MaxPooling2D(pool_size=2, strides=2))
print("First Conv2D and MaxPooling2D layers added successfully!")

# Second Convolutional layer
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
print("Second Conv2D and MaxPooling2D layers added successfully!")

# Flattening layer
# Flattening converts the 2D matrix into a 1D vector to prepare for the fully connected layers
cnn.add(Flatten())
print("Flattening layer added successfully!")

'''
Fully connected Dense layer:

units is set to 128 to have 128 neurons in this layer
activation is set to 'relu' to introduce non-linearity
'''
cnn.add(Dense(units=128, activation='relu'))
print("Dense layer added successfully!")

# Output layer
# units is set to 1 for binary classification
# activation is set to 'sigmoid' to output probabilities between 0 and 1
cnn.add(Dense(units=1, activation='sigmoid'))
print("Output layer added successfully!")

'''
Compile the CNN:

optimizer is set to 'adam' for efficient training
loss is set to 'binary_crossentropy' for binary classification tasks
metrics is set to 'accuracy' to monitor the accuracy during training
'''
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("CNN compiled successfully!")

# Train the CNN
# epochs is set to 25 to train the model for 25 iterations over the dataset
cnn.fit(x=training_set, validation_data=test_image_gen, epochs=25)
print("CNN training completed successfully!")

# Single image prediction
# target_size is set to (64, 64) to match the input size of the CNN
test_image = load_img(path="Part 2 - CNN - Convolutional Neural Network\\example_datasets\\single_prediction\\cat_or_dog_3.jpg", target_size=(64, 64))

# Convert image to array
test_image = img_to_array(test_image)

# Expand dimensions to match the input shape of the CNN
test_image = np.expand_dims(test_image, axis=0)

# Make prediction and prints raw result value
result = cnn.predict(test_image)
print("Raw prediction: ", result)

# Makes prediction based on maximum value from the prediction
predicted_index = np.argmax(result[0])
prediction = class_names[predicted_index]

# If prediction is greater than 0.5, output dog, else output cat
print("Prediction: ", prediction)
