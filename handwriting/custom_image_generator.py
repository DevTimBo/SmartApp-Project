# Author: Jason Pranata
# Final Version: 13 February 2023

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CustomImageGenerator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Values have been optimized within reason, if processing too slow, then take some augmentations out
        # Augmentations were chosen to emulate different writing styles, of course within the module's capability
        self.image_data_generator = ImageDataGenerator(
            rotation_range=0.5,
            shear_range=1,
            zoom_range=0.025,
            width_shift_range=0.1,
            height_shift_range=0.005,
            rescale=1./255,
            fill_mode='nearest',
        )

    def generate_augmented_batch(self):
        for batch in self.dataset:
            images, labels = batch["image"], batch["label"]

            # Generate augmented images
            augmented_images = []
            for img in images:
                augmented_img = self.image_data_generator.random_transform(img.numpy())
                augmented_images.append(augmented_img)

            augmented_images = np.array(augmented_images)

            yield {'image': augmented_images, 'label': labels}, np.array(labels)

    def generator(self):
        while True:
            yield from self.generate_augmented_batch()
