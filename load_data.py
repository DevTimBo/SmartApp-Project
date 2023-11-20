# Imports
from keras.layers import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)
base_path = "data"
lines_list = []




def read_data():
    global lines_list
    # Read the file with UTF-8 encoding
    with open(f"{base_path}/lines.txt", "r", encoding="utf-8") as file:
        words = file.readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            lines_list.append(line)

    print(f"Dataset contains: {len(lines_list)} lines")

    np.random.shuffle(lines_list)


train_samples, test_samples = 0, 0


def split_data():
    global train_samples, test_samples
    global lines_list
    split_idx = int(0.9 * len(lines_list))
    train_samples = lines_list[:split_idx]
    test_samples = lines_list[split_idx:]

    print(f"Total train samples: {len(train_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    return train_samples, test_samples



base_image_path = os.path.join(base_path, "lines")


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels, test_img_paths, test_labels = 0, 0, 0 ,0


def create_train_and_test_data():
    global train_img_paths, train_labels, test_img_paths, test_labels
    global test_samples, train_samples
    test_samples, train_samples = split_data()
    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)


max_len = 0

train_labels_cleaned = 0
def get_vocabulary_length_and_clean_labels():
    # Find maximum length and the size of the vocabulary in the training data.
    global train_labels_cleaned
    global train_labels
    train_labels_cleaned_intern = []
    characters = set()
    global max_len

    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        train_labels_cleaned_intern.append(label)

    train_labels_cleaned = train_labels_cleaned_intern
    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))

    # Check some label samples.
    print("Labels Cleaned:")
    print(train_labels_cleaned[:10])

test_labels_cleaned = 0
def clean_test_labels():
    global test_labels
    global test_labels_cleaned
    cleaned_labels = []
    for label in test_labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    test_labels_cleaned = cleaned_labels


