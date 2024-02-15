#5010890
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

def resize_images_in_directory(source_dir, target_dir, new_size=(2480, 3507)):
    """
    Resize all JPEG images in a directory and save them to a target directory.
    2480x3507
    :param source_dir: Directory containing the original images.
    :param target_dir: Directory where the resized images will be saved.
    :param new_size: Tuple of (width, height) for the new size.
    """
    # Create the target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Get all JPEG files in the source directory
    jpeg_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]

    # Loop through all JPEG files in the source directory with a progress bar
    for file_name in tqdm(jpeg_files, desc="Resizing Images"):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        with Image.open(source_path) as img:
            # Resize the image
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save the resized image
            resized_img.save(target_path, quality=95, optimize=True)

# Example usage
#resize_images_in_directory('rotated', 'resized')
