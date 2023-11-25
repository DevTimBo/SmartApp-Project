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


show_image(x_train[1])
show_image_w_path(x_train_img_paths[1])


