from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Thresholding
    threshold = 200
    binary_image = grayscale_image.point(lambda p: p > threshold and 255)

    # Invert the image
    inverted_image = ImageOps.invert(binary_image)

    # Dilation to make the text thicc
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(np.array(inverted_image), kernel, iterations=1)
    
    # Convert the image back
    dilated_image = Image.fromarray(dilated_image)

    return dilated_image

def is_checkbox_checked(image_path):
    # Preprocess
    processed_image = preprocess_image(image_path)

    # Get the width and height of the image
    width, height = processed_image.size

    # Define the ja and nein regions 
    ja_region = (0, 0, width // 2, height)
    nein_region = (width // 2, 0, width, height)

    # Crop the image to get the regions
    ja_image = processed_image.crop(ja_region)
    nein_image = processed_image.crop(nein_region)

    # Plot the processed image and the cropped regions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Processed Image")

    plt.subplot(1, 3, 2)
    plt.imshow(ja_image, cmap="gray")
    plt.title("Cropped Ja Region")

    plt.subplot(1, 3, 3)
    plt.imshow(nein_image, cmap="gray")
    plt.title("Cropped Nein Region")

    plt.show()

    # Count the black pixels in the cropped regions
    ja_x_count = ja_image.tobytes().count(b'\x00')
    nein_x_count = nein_image.tobytes().count(b'\x00')
    print(f"Ja X Count: {ja_x_count}", f"Nein X Count: {nein_x_count}")

    # Determine the result based on which one has more black pixels
    if ja_x_count > nein_x_count:
        checkbox_result = "Nein"
    elif nein_x_count > ja_x_count:
        checkbox_result = "Ja"
    else:
        checkbox_result = "Unknown"
    
    return checkbox_result

# Another possibility but probably not as good
def is_checkbox_checked2(image):
    # Preprocess
    processed_image = preprocess_image(image)

    # Plot the processed image and the cropped regions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Processed Image")

    plt.show()
    
    checked_count = processed_image.tobytes().count(b'\x00')
    print(f"Checked Count: {checked_count}")
    
    # Determine the result based on threshold of black pixels
    threshold = 1300
    if checked_count > threshold:
        checkbox_result = "Ja"
    elif threshold > checked_count:
        checkbox_result = "Nein"
    else:
        checkbox_result = "Unknown"
    
    return checkbox_result

# Example usage
image_path = "contrast_true_or_false\croppedja3.png"
# Load the image
image = Image.open(image_path)
result = is_checkbox_checked(image)
print(f"The checkbox is checked for: {result}")
