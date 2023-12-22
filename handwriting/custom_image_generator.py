import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CustomImageGenerator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

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

    def generate_augmented_batch(self):
        for batch in self.dataset:
            images, labels = batch["image"], batch["label"]

            # Generate augmented images
            augmented_images = []
            for img in images:
                #print(img.shape)
                augmented_img = self.image_data_generator.random_transform(img.numpy())
                augmented_images.append(augmented_img)

            augmented_images = np.array(augmented_images)

            yield {'image': augmented_images, 'label': labels}, np.array(labels)

    def generator(self):
        while True:
            yield from self.generate_augmented_batch()


''' # To visualize augmentations
# To see the augmentations from CustomImageGenerator
train_generator = cgi.CustomImageGenerator(train_ds, BATCH_SIZE)
example_batch = train_generator.generator().__next__()
augmented_images = example_batch[0]['image']

num_to_plot = 4
fig, axes = plt.subplots(1, num_to_plot, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.squeeze(augmented_images[i]), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
'''
