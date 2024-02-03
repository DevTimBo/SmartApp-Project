from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
plotting = True
def detect_and_plot_bounding_boxes(image, keywords):
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Resize images
    image = cv2.resize(image, (512, 128))
    
    # Convert the numpy array back to PIL image
    pil_image = Image.fromarray(image)
    
    # Perform OCR
    ocr_results = pytesseract.image_to_boxes(pil_image, lang='eng')

    # Initialize a list to store the cropped images
    cropped_images = []

    # Draw the bounding boxes on the original image
    draw = ImageDraw.Draw(pil_image)

    # Loop through the OCR results
    for line in ocr_results.splitlines():
        # Split the line into parts (character, x, y, width, height, etc.)
        parts = line.split()

        # Check if the recognized text is one of the keywords
        if parts[0] in keywords:
            # Extract coordinates of the bounding box-- we just need x
            x, y, w, h = map(int, (parts[1], parts[2], parts[3], parts[4]))

            # Draw the bounding box on the original image
            #draw.rectangle([x, image.height - (y + h), x + w, image.height - y], outline="red", width=2)

            # Crop the image based on the bounding box
            cropped_image_ja = image.crop((0, 0, x + 20, image.height))
            cropped_image_nein = image.crop((x + 20, 0, image.width, image.height))

            # Append the cropped image to the list
            cropped_images.append(cropped_image_ja)
            cropped_images.append(cropped_image_nein)
    if plotting:
        # Plot the original image with bounding boxes
        plt.imshow(image)
        plt.title("Original Image with Bounding Boxes")
        plt.show()

        # Plot the cropped images
        for i, cropped_image in enumerate(cropped_images):
            plt.subplot(1, len(cropped_images), i + 1)
            plt.imshow(cropped_image)
            plt.title(f"Cropped Image {i + 1}")

        plt.show()

    if cropped_images != []:
        return cropped_images[0], cropped_images[1]
    else:
        return None, None

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

def is_checkbox_checked(image, PLOTTING):
    global plotting
    plotting = PLOTTING

    keywords = ['a']
    ja_image, nein_image = detect_and_plot_bounding_boxes(image, keywords)
    if ja_image != None and nein_image != None:
        ja_image = preprocess_image(ja_image)
        nein_image = preprocess_image(nein_image)
    #Exception handling basically    
    else:
        processed_image = preprocess_image(image)
        width, height = processed_image.size
        # Define the ja and nein regions 
        ja_region = (0, 0, width // 2, height)
        nein_region = (width // 2, 0, width, height)

        # Crop the image to get the regions
        ja_image = processed_image.crop(ja_region)
        nein_image = processed_image.crop(nein_region)

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

def is_checkbox_checked_with_plot(image):    
    keywords = ['a']
    ja_image, nein_image = detect_and_plot_bounding_boxes(image, keywords)
    if ja_image != None and nein_image != None:
        ja_image = preprocess_image(ja_image)
        nein_image = preprocess_image(nein_image)
    #Exception handling basically    
    else:
        processed_image = preprocess_image(image)
        width, height = processed_image.size
        # Define the ja and nein regions 
        ja_region = (0, 0, width // 2, height)
        nein_region = (width // 2, 0, width, height)

        # Crop the image to get the regions
        ja_image = processed_image.crop(ja_region)
        nein_image = processed_image.crop(nein_region)
    if plotting:
        # Plot the processed image and the cropped regions
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Image")

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

def is_checkbox_checked_nur_ja(image):
    # Preprocess
    processed_image = preprocess_image(image)
    if plotting:
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

def is_checkbox_checked_template(checkbox_image):
    # Resize images
    checkbox_image = cv2.resize(checkbox_image, (512, 128))
    
    template_path = 'contrast_true_or_false/templatebox1.png'
    template = cv2.imread(template_path)
    template = cv2.resize(template, (100, 100))

    # Convert images to grayscale
    checkbox_gray = cv2.cvtColor(checkbox_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(checkbox_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get the coordinates of the matched region
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the matched region
    matched_image = checkbox_image.copy()
    cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)
    if plotting:
        # Plot the images and the matched region
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 3, 1), plt.imshow(checkbox_image, cmap='gray')
        plt.title('Checkbox Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2), plt.imshow(template, cmap='gray')
        plt.title('Template Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 3), plt.imshow(matched_image, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

        plt.show()

    # Check if the matching result is above the threshold
    print(max_val)
    if max_val > 0.7:
        checkbox_checked = "Ja"
    else:
        checkbox_checked = "Nein"

    # Return the result
    return checkbox_checked

'''
# Example usage single checkbox
checkbox_image_path = 'contrast_true_or_false/ja_unchecked_middle.png'
# Load images
checkbox_image = cv2.imread(checkbox_image_path)
result = is_checkbox_checked_template(checkbox_image)
print(f"The checkbox is: {result}")


# Example usage tof
image_path = 'contrast_true_or_false\cropped2.png'
# Load the image
image = Image.open(image_path)
result = is_checkbox_checked_with_plot(image)
print(f"The checkbox is checked for: {result}")
'''