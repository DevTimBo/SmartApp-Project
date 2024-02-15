#Autoren Tristan und Alireza
import cv2
import numpy as np
import os

from bounding_box.config import YOLO_HEIGHT, YOLO_WIDTH


def calculate_bbox_scale_factor(old_bbox, new_bbox):
    # Calculate the scale factors to resize bounding boxes from an old size to a new size.
    # Extract the last bounding box from the list (assuming it contains the desired bounding box)
    old_bbox = old_bbox[(len(old_bbox) - 1)]
    # Calculate the width and height of the old bounding box
    old_width = old_bbox[2] - old_bbox[0]
    old_height = old_bbox[3] - old_bbox[1]
    # Calculate the width and height of the new bounding box
    new_width = new_bbox[2] - new_bbox[0]
    new_height = new_bbox[3] - new_bbox[1]
    # Calculate the scale factors for width and height
    width_scale_factor = new_width / old_width
    height_scale_factor = new_height / old_height

    return width_scale_factor, height_scale_factor


def resize_imaged_without_expand_dim(image, width, height):
    # resize image but not change it to numpy array
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def cut_links_bbox(verkleinerungsfaktor, bbox):
    # Calculate the new x-coordinate of the left side of a bounding box after cutting it
    # by a certain percentage from the left side.
    return bbox[0] + (bbox[2] - bbox[0]) * (1 - verkleinerungsfaktor)


def cut_top_bbox(verkleinerungsfaktor, bbox):
    # Calculate the new y-coordinate of the left side of a bounding box after cutting it
    # by a certain percentage from the top side.
    return bbox[1] + (bbox[3] - bbox[1]) * verkleinerungsfaktor


def cut_rechts_bbox(verkleinerungsfaktor, bbox):
    # Calculate the new x-coordinate of the right side of a bounding box after cutting it
    # by a certain percentage from the right side.
    return bbox[2] - (bbox[2] - bbox[0]) * verkleinerungsfaktor


def add_top_bbox(vergroesserungsfaktor, bbox):
    #  Calculate the new y-coordinate of the top side of a bounding box after adding a certain percentage to it.
    return bbox[1] - (bbox[3] - bbox[1]) * vergroesserungsfaktor


def add_bottom_bbox(vergroesserungsfaktor, bbox):
    # Calculate the new y-coordinate of the bottom side of a bounding box after adding a certain percentage to it.
    return bbox[3] + (bbox[3] - bbox[1]) * vergroesserungsfaktor


def add_rechts_bbox(vergroesserungsfaktor, bbox):
    # Calculate the new x-coordinate of the right side of a bounding box after adding a certain percentage to it.
    return bbox[2] + (bbox[2] - bbox[0]) * vergroesserungsfaktor


def add_links_bbox(vergroesserungsfaktor, bbox):
    # Calculate the new x-coordinate of the link side of a bounding box after adding a certain percentage to it.
    return bbox[0] - (bbox[2] - bbox[0]) * vergroesserungsfaktor


def get_position_difference_between_boxes(new_bbox, old_bbox):
    # Calculate the position difference between two bounding boxes.
    xmin = new_bbox[0] - old_bbox[0]
    ymin = new_bbox[1] - old_bbox[1]
    xmax = new_bbox[2] - old_bbox[2]
    ymax = new_bbox[3] - old_bbox[3]
    return xmin, ymin, xmax, ymax


def adjust_position_of_the_boxes(xmindiff, ymindiff, xmaxdiff, ymaxdiff, boxes):
    # Adjust the position of multiple bounding boxes based on the provided differences.
    adjusted_boxes = []
    # Iterate through each bounding box
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        # Adjust coordinates based on the provided differences
        xmin -= xmindiff
        ymin -= ymindiff
        xmax -= xmaxdiff
        ymax -= ymaxdiff

        # Add the adjusted bounding box to the list
        adjusted_box = [xmin, ymin, xmax, ymax]
        adjusted_boxes.append(adjusted_box)
    return adjusted_boxes


def resize_image(image_path, image_width, image_height):
    # Read the image from the specified path
    image = cv2.imread(image_path)
    # Resize the image to the specified width and height
    resized_image = cv2.resize(image, (image_width, image_height))
    # Add an extra dimension to make it compatible with model input
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image

def get_width_height_shape(image_path):
    #Calculate the width and height ratios required to resize the image to fit within specified dimensions.
    ## Read the image from the specified path
    image = cv2.imread(image_path)
    # Check if the image dimensions are larger than the YOLO_WIDTH and YOLO_HEIGHT
    if image.shape[0] > YOLO_HEIGHT and image.shape[1] > YOLO_WIDTH:
        # if image size ist gratter than yolo size
        height_ratio = image.shape[0] / YOLO_HEIGHT
        width_ratio = image.shape[1] / YOLO_WIDTH
    else:
        # if image size is smaller than yolo-size
        height_ratio = YOLO_HEIGHT / image.shape[0]
        width_ratio = YOLO_WIDTH / image.shape[1]
    return width_ratio, height_ratio


def scale_bounding_one_box(bbox, width_ratio, height_ratio):
    # Scale a single bounding box based on the width and height ratios
    x_min = np.round(bbox[0] * width_ratio, 2)
    y_min = np.round(bbox[1] * height_ratio, 2)
    x_max = np.round(bbox[2] * width_ratio, 2)
    y_max = np.round(bbox[3] * height_ratio, 2)

    return [x_min, y_min, x_max, y_max]


def scale_bounding_box(bounding_boxes, width_ratio, height_ratio):
    # Scale a list of bounding boxes based on the width and height ratios.
    scalde_bounding_boxes = []

    for box in bounding_boxes:
        x_min = np.round(box[0] * width_ratio, 2)
        y_min = np.round(box[1] * height_ratio, 2)
        x_max = np.round(box[2] * width_ratio, 2)
        y_max = np.round(box[3] * height_ratio, 2)
        # Add the scaled bounding box to the list
        scalde_bounding_boxes.append([x_min, y_min, x_max, y_max])
    return scalde_bounding_boxes


def scale_up(ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links, ratios):
    # Scale up the bounding boxes for different classes based on the calculated ratios.

    # Iterate through each bounding box of Ausbildung and adjust the size
    for i, box, cls in zip(range(len(ausbildung_cut_links[1])), ausbildung_cut_links[0], ausbildung_cut_links[1]):
        ausbildung_cut_links[0][i] = scale_bounding_one_box(box, ratios[0], ratios[1])

    # Iterate through each bounding box of person and adjust the size
    for i, box, cls in zip(range(len(person_cut_links[1])), person_cut_links[0], person_cut_links[1]):
        person_cut_links[0][i] = scale_bounding_one_box(box, ratios[0], ratios[1])

    # Iterate through each bounding box of wohnsitz and adjust the size
    for i, box, cls in zip(range(len(wohnsitz_cut_links[1])), wohnsitz_cut_links[0], wohnsitz_cut_links[1]):
        wohnsitz_cut_links[0][i] = scale_bounding_one_box(box, ratios[0], ratios[1])

    # Iterate through each bounding box of wwa and adjust the size
    for i, box, cls in zip(range(len(wwa_cut_links[1])), wwa_cut_links[0], wwa_cut_links[1]):
        wwa_cut_links[0][i] = scale_bounding_one_box(box, ratios[0], ratios[1])

    return ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links
