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

def prepare_augmented_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        lambda x, y: (process_images_labels(x, y)["image"], process_images_labels(x, y)["label"]),
        num_parallel_calls=AUTOTUNE
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(1, seed=42),
        tf.keras.layers.RandomBrightness(1, value_range=(0, 1), seed=42)
    ])
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Separate image and label
    dataset = dataset.map(lambda x, y: {"image": x, "label": y}, num_parallel_calls=AUTOTUNE)

    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

