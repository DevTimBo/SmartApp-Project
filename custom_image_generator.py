import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import tokenizer

class CustomImageGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_width, image_height):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_samples = len(image_paths)

        # Values have been optimized within reason, if processing too slow, then take some augmentations out
        self.image_data_generator = ImageDataGenerator(
            rotation_range=0.5,
            shear_range=1,
            zoom_range=0.025,
            width_shift_range=0.1,
            height_shift_range=0.005,
            rescale=1./255, 
            fill_mode='nearest',
        )

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

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