from keras.layers import StringLookup
import tensorflow as tf
import load_data
import preprocess
import numpy as np
AUTOTUNE = tf.data.AUTOTUNE
max_len = load_data.max_len
padding_token = 99
batch_size = 64


# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(load_data.characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def process_images_labels(image_path, label):
    image = preprocess.preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_data(image_paths, labels):
    processed_data = [process_images_labels(image_path, label) for image_path, label in zip(image_paths, labels)]

    # Separate the processed data into x_train and y_train
    x_train = np.array([item['image'] for item in processed_data])
    y_train = np.array([item['label'] for item in processed_data])

    return x_train, y_train

