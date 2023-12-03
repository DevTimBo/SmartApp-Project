import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import tokenizer
import tensorflow as tf
import os
from PIL import Image

base_path = 'data'
aug_img_path = os.path.join(base_path, "augmented")

def save_augmented_image(img, base_path=base_path):
    # Create the directory if it doesn't exist
    os.makedirs(aug_img_path, exist_ok=True)

    # Set unique filename
    unique_filename = f"augmented_image_{np.random.randint(100000)}.png"

    # Concatenate the img_path and unique filename to get the full path
    full_path = os.path.join(aug_img_path, unique_filename)

    # Convert the array to an image and save
    augmented_image = Image.fromarray(np.uint8(img))
    augmented_image.save(full_path)

    return full_path

class CustomImageGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_width, image_height):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_samples = len(image_paths)
        
        self.image_data_generator = ImageDataGenerator(
            rotation_range=0.2,
            shear_range=0.5,
            zoom_range=0.1, # Adjust for stretching
            #brightness_range=[0.6, 1.3],
            #width_shift_range=0.05,
            #height_shift_range=0.05,
            fill_mode='nearest',
            #horizontal_flip=True,
            #vertical_flip=False,
            rescale=1./255
        )

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    # This one is for merging datasets:
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Prepare Data
        batch_image_paths, batch_labels = self.image_paths[start_index:end_index], self.labels[start_index:end_index]
    
        # Generate augmented image with their paths
        augmented_image_paths = []
        for image_path in batch_image_paths:
            img = tf.keras.preprocessing.image.load_img(image_path)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.image_data_generator.random_transform(img)
            augmented_image_path = save_augmented_image(img) 
            augmented_image_paths.append(augmented_image_path)
    
        return {'image': augmented_image_paths, 'label': batch_labels}, np.array(batch_labels)
    '''
    # The following works for training with augmented data or for plots:
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Prepare Data
        batch_images, padded_labels = tokenizer.prepare_data(self.image_paths[start_index:end_index], self.labels[start_index:end_index])
    
        # Generate augmented images
        augmented_images = []
        for img in batch_images:
            img = self.image_data_generator.random_transform(img)
            augmented_images.append(img)
    
        augmented_images = np.array(augmented_images)

        return {'image': augmented_images, 'label': padded_labels}, np.array(padded_labels)
    '''

def merge_datasets(custom_generator, dataset):
    #for round in range(rounds): #Loop if we want to make and merge more than 1 augmented dataset
    for batch_index in range(len(custom_generator)):
        augmented_image_paths, batch_labels = custom_generator[batch_index]

        # Create data set with augmented images
        augmented_ds = tokenizer.prepare_dataset(augmented_image_paths['image'], batch_labels)

        # Merge original and augmented datasets
        dataset = dataset.concatenate(augmented_ds)
            
    return dataset