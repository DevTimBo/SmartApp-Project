from keras.preprocessing import image

BATCH_SIZE = 32

def init_image_generator(sample_image, batch_size=BATCH_SIZE):
    datagen = image.ImageDataGenerator(
        rotation_range=0.2,
        shear_range=0.5,
        zoom_range=0.05,
        rescale=1./255
    )

    # Reshape the image
    sample_image = sample_image.reshape((1,) + sample_image.shape)

    return datagen.flow(sample_image, batch_size=batch_size)

def sample_image(dataset):
    for batch in dataset.take(1):  
        # Select the first image from the batch
        sample_image = batch['image'][0].numpy()
    
    return sample_image