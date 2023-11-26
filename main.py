import load_data
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

load_data.read_data()
load_data.split_data()
load_data.create_train_and_test_data()
load_data.get_vocabulary_length_and_clean_labels()
load_data.clean_test_labels()

x_train_img_paths, y_train_labels = load_data.train_img_paths, load_data.train_labels_cleaned
x_test_img_paths, y_test_labels = load_data.test_img_paths, load_data.test_labels_cleaned


# Has to be here because load data functions need to be called before
import tokenizer
import preprocess

x_train, y_train = tokenizer.prepare_data(x_train_img_paths, y_train_labels)
x_test, y_test = tokenizer.prepare_data(x_train_img_paths, y_train_labels)
def show_image(img):
    # Ensure that the image values are in the range [0, 255]
    img = (img * 255).astype(np.uint8)

    # Display the image using OpenCV
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def show_image_w_path(image_path):
    # Read the image from file
    image = cv.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        # Display the image
        cv.imshow('Image', image)
        cv.waitKey(0)  # Wait until a key is pressed
        cv.destroyAllWindows()  # Close the window


num_classes = len(load_data.characters)
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, MaxPooling2D, Reshape

# Define the input shape for images
image_shape = (512, 32, 1)

# Define the model
model = Sequential()

# Add a Conv2D layer before LSTM layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Reshape the output of Conv2D layer to fit LSTM input
model.add(Reshape((-1, 32 * 128)))

# Encoder
model.add(LSTM(256, return_sequences=True))

# Decoder
model.add(LSTM(256, return_sequences=True))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

