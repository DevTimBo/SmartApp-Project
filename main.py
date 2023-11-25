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

train_ds = tokenizer.prepare_dataset(x_train_img_paths, y_train_labels)
test_ds = tokenizer.prepare_dataset(x_test_img_paths, y_test_labels)
def show_image(img):
    # Convert the image to uint8 if it's in a different data type
    img = img.astype(np.uint8)

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
def show_dataset(dataset, num_images=4):
    for data in dataset.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(2, 2, figsize=(10, 8))

        for i in range(min(num_images, 4)):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(tf.math.not_equal(label, tokenizer.padding_token)))
            # Convert to string.
            label = tf.strings.reduce_join(tokenizer.num_to_char(indices))
            label = label.numpy().decode("utf-8")

            ax[i // 2, i % 2].imshow(img, cmap="gray")
            ax[i // 2, i % 2].set_title(label)
            ax[i // 2, i % 2].axis("off")

    plt.show()



show_dataset(train_ds)
show_image_w_path(x_train_img_paths[1])


