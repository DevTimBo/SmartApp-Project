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
        
        self.image_data_generator = ImageDataGenerator(
            rotation_range=0.2,
            shear_range=0.5,
            zoom_range=0.05, # Adjust for stretching
            #width_shift_range=0.5, #Probably not needed
            #height_shift_range=0.5,
            rescale=1./255
        )

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Prepare Data
        batch_images, padded_labels = tokenizer.prepare_data(self.image_paths[start_index:end_index], self.labels[start_index:end_index])
        augmented_images = self.image_data_generator.flow(batch_images, shuffle=False).next()

        return {'image': augmented_images, 'label': padded_labels}, np.zeros(self.batch_size)