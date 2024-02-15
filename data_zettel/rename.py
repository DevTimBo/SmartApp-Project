#5010890
import os
from pathlib import Path

def rename_images_in_directory(directory):
    
    # Get all image files in the directory, sorted alphabetically
    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.jpg')])

    # Loop through all image files and rename them
    for i, file_name in enumerate(image_files, start=1):
        old_path = Path(directory) / file_name
        new_name = f"image_{i:04d}.jpg"
        new_path = Path(directory) / new_name

        os.rename(old_path, new_path)

# Execute
rename_images_in_directory('resized')