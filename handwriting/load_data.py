# Authors: Tim Harmling and Alexej Kravtschenko
# Loads in the data from the IAM Dataset Path where
# the data is stored can be changed in the config file usually its located in /data/

import numpy as np
import os
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

characters = set()
max_len = 0
base_path = "./data"  # gets overwritten by config
base_image_path = os.path.join(base_path, "lines")


def read_data():
    lines_list = []
    # Read the file with UTF-8 encoding
    with open(f"{base_path}/lines.txt", "r", encoding="utf-8") as file:
        words = file.readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            lines_list.append(line)
    
    np.random.shuffle(lines_list)
    return lines_list

def split_data(lines_list):
    split_idx = int(0.9 * len(lines_list))
    train_samples = lines_list[:split_idx]
    test_samples = lines_list[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    return train_samples, test_samples, validation_samples


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for _, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


def get_vocabulary_length_and_clean_labels(train_labels):
    train_labels_cleaned = []
    global characters, max_len

    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        train_labels_cleaned.append(label)

    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return train_labels_cleaned


def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


data = read_data()
train_samples, test_samples, validation_samples = split_data(data)

def print_samples(new_base_path):
    global base_path
    base_path = new_base_path
    print(f"Total train samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")
    
def get_train_data():
    train_path, train_label = get_image_paths_and_labels(train_samples)
    train_labels_cleaned = get_vocabulary_length_and_clean_labels(train_label)
    
    return train_path, train_labels_cleaned

def get_validation_data():
    val_path, val_label = get_image_paths_and_labels(validation_samples)
    validation_labels_cleaned = clean_labels(val_label)
    
    return val_path, validation_labels_cleaned

def get_test_data():
    test_path, test_label = get_image_paths_and_labels(test_samples)
    test_labels_cleaned = clean_labels(test_label)
    
    return test_path, test_labels_cleaned
    
