import cv2
import numpy as np
import os

from config import YOLO_HEIGHT, YOLO_WIDTH

def get_image_size(image):
    height= image.shape[0]
    width = image.shape[1]
    return width, height

def resize_image(image_path, image_width, image_height):
    image =  cv2.imread(image_path)
    resized_image = cv2.resize(image, (image_width, image_height))
    resized_image = np.expand_dims(image, axis=0)  
    return resized_image 

def resize(input_path, output_path, width, height):
    image =  cv2.imread(input_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(output_path, resized_image)    

def get_width_height_shape(image_path):
    image =  cv2.imread(image_path)
    height_ratio = YOLO_HEIGHT / image.shape[0]
    width_ratio = YOLO_WIDTH / image.shape[1]
    return width_ratio, height_ratio

def scale_bounding_box(bounding_boxes, width_ratio, height_ratio):
    scalde_bounding_boxes = []
    
    for box in bounding_boxes:
        x_min = np.round(box[0]*width_ratio,2)
        y_min = np.round(box[1]*height_ratio,2)
        x_max = np.round(box[2]*width_ratio,2)
        y_max = np.round(box[3]*height_ratio,2)
        scalde_bounding_boxes.append([x_min, y_min, x_max, y_max])
    return scalde_bounding_boxes

def create_directories(output_path_images, output_path_annotations, new_width, new_height):
    output_path_images = output_path_images + '/' + str(new_height) + 'x' + str(new_width)
    output_path_annotations = output_path_annotations + '/' + str(new_height) + 'x' + str(new_width)
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_annotations, exist_ok=True)
