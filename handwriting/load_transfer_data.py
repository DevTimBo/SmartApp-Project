import numpy as np
import os
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

characters = set()
max_len = 0

import os
cwd = os.getcwd()
last_part = os.path.basename(cwd)

if last_part == "SmartApp-Project":
    base_path = "data_zettel/cropped_images/"  # path for pipeline.py
else:
    base_path = "../data_zettel/cropped_images/"  # path for handwriting

def read_data():
    data_list = []
    image_files = [f for f in os.listdir(base_path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]

        img_path = os.path.join(base_path, image_file)
        label_file = os.path.join(base_path, f"{image_name}.txt")

        if os.path.exists(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as file:
                    print(image_name)

                    line = file.readline().strip()

            except UnicodeDecodeError as e:
                continue
            data_list.append((img_path, line))


    #np.random.shuffle(data_list) # Rausgenommen zum testen
    return data_list

def split_data(lines_list):
    split_idx = int(0.9 * len(lines_list))
    train_samples = lines_list[:split_idx]
    test_samples = lines_list[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    return train_samples, test_samples, validation_samples


def get_image_paths_and_labels(samples):
    x_img_paths = []
    y_labels = []

    for img_path, label in samples:
        if os.path.exists(img_path):
            x_img_paths.append(img_path)
            y_labels.append(label)

    return x_img_paths, y_labels


def get_vocabulary_length(data):
    global characters, max_len

    for _, label in data:
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))

    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return characters, max_len


data = read_data()
all_data = read_data()
characters, max_len = get_vocabulary_length(all_data)
train_samples, test_samples, validation_samples = split_data(data)


def print_samples():
    print(f"Total train samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")
    
def get_train_data():
    train_path, train_label = get_image_paths_and_labels(train_samples)

    
    return train_path, train_label

def get_validation_data():
    val_path, val_label = get_image_paths_and_labels(validation_samples)

    
    return val_path, val_label

def get_test_data():
    test_path, test_label = get_image_paths_and_labels(test_samples)

    
    return test_path, test_label
    
