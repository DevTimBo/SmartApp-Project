from keras.layers import StringLookup
import tensorflow as tf

import preprocess
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
padding_token = 99
batch_size = 64

import os
import pickle

with open("vocab_max_len.pkl", "rb") as file:
    loaded_vocab, loaded_max_len = pickle.load(file)


characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']
max_len = loaded_max_len

char_to_num = StringLookup(vocabulary=sorted(characters), mask_token=None)

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

def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

def prepare_data(image_paths, labels):
    processed_data = [process_images_labels(image_path, label) for image_path, label in zip(image_paths, labels)]

    # Separate the processed data into x_train and y_train
    x_train = np.array([item['image'] for item in processed_data])
    y_train = np.array([item['label'] for item in processed_data])

    return x_train, y_train

