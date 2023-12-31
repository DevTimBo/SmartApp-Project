from keras.layers import StringLookup
import tensorflow as tf
import load_data
import preprocess
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
max_len = load_data.max_len

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(load_data.characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

img_size = (512, 32)  # default gets overwritten by config
batch_size = 64  # default gets overwritten by config
padding_token = 99


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess.preprocess_image(image_path, img_size)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, img_size_config, batch_size_new):
    global img_size
    img_size = img_size_config
    global batch_size
    batch_size = batch_size_new
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

def prepare_augmented_dataset(image_paths, labels, batch_size_new):
    #Prepare Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        lambda x, y: (process_images_labels(x, y)["image"], process_images_labels(x, y)["label"]),
        num_parallel_calls=AUTOTUNE
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.5, seed=42),
        tf.keras.layers.RandomBrightness(0.5, value_range=(0, 1), seed=42)
    ])
    #Apply Augmentation
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Separate image and label
    dataset = dataset.map(lambda x, y: {"image": x, "label": y}, num_parallel_calls=AUTOTUNE)

    return dataset.batch(batch_size_new).cache().prefetch(AUTOTUNE)

