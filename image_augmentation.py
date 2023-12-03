from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
#Deprecated - use to test augmentations

def init_image_generator(sample_image):#, sample_label):
    datagen = image.ImageDataGenerator(
        rotation_range=0.1,
        shear_range=1,
        zoom_range=0..05,
        rescale=1./255
    )

    # Reshape
    #sample_image = sample_image.reshape((1,) + sample_image.shape)
    
    #sample_image = sample_image.squeeze()
    sample_image = np.expand_dims(sample_image, axis=0)
    #sample_label = np.expand_dims(sample_label, axis=0)
    
    #sample_label = sample_label.squeeze()
    #return datagen.flow(sample_image)
    return datagen.flow(sample_image) #, sample_label)

def sample_image(dataset):
    for batch in dataset.take(1):  
        # Select the first image from the batch
        sample_image = batch['image'][0].numpy()
        sample_label = batch['label'][0].numpy()
    
    return sample_image #, sample_label