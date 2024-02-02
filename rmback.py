import cv2
import numpy as np
import os

def isolate_paper(input_image_path, output_image_path):
  
    
   
    # Read the input image
    original_image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # Find the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(max(large_contours, key=cv2.contourArea))

    # Crop the region of interest (ROI) from the original image
    isolated_paper = original_image[y:y+h, x:x+w]

    # Save the isolated paper as a new image
    cv2.imwrite(output_image_path, isolated_paper)
    return isolated_paper

