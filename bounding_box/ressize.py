import cv2
import numpy as np
import os

from bounding_box.config import YOLO_HEIGHT, YOLO_WIDTH


def calculate_bbox_scale_factor(old_bbox, new_bbox):
    old_bbox = old_bbox[(len(old_bbox) - 1)]

    old_width = old_bbox[2] - old_bbox[0]
    old_height = old_bbox[3] - old_bbox[1]

    new_width = new_bbox[2] - new_bbox[0]
    new_height = new_bbox[3] - new_bbox[1]

    width_scale_factor = new_width / old_width
    height_scale_factor = new_height / old_height

    return width_scale_factor, height_scale_factor


def scale_box(new_image_width, new_image_height, box):
    width_scale = new_image_width / 640
    height_scale = new_image_height / 640
    xmin, ymin, xmax, ymax = box
    # Bbox-Koordinaten anpassen
    new_xmin = int(xmin * width_scale)
    new_ymin = int(ymin * height_scale)
    new_xmax = int(xmax * width_scale)
    new_ymax = int(ymax * height_scale)

    return new_xmin, new_ymin, new_xmax, new_ymax


def resize_imaged_without_expand_dim(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def cut_links_bbox(verkleinerungsfaktor, bbox):
    return bbox[0] + (bbox[2] - bbox[0]) * (1 - verkleinerungsfaktor)


def cut_top_bbox(verkleinerungsfaktor, bbox):
    return bbox[1] + (bbox[3] - bbox[1]) * verkleinerungsfaktor


def add_bottom_bbox(vergroesserungsfaktor, bbox):
    return bbox[3] + (bbox[3] - bbox[1]) * vergroesserungsfaktor


def add_rechts_bbox(vergroesserungsfaktor, bbox):
    return bbox[2] + (bbox[2] - bbox[0]) * vergroesserungsfaktor


def cut_rechts_bbox(vergroesserungsfaktor, bbox):
    return bbox[2] - (bbox[2] - bbox[0]) * vergroesserungsfaktor


def add_links_bbox(vergroesserungsfaktor, bbox):
    return bbox[0] - (bbox[2] - bbox[0]) * vergroesserungsfaktor


def get_position_difference_between_boxes(new_bbox, old_bbox):
    # new - old
    xmin = new_bbox[0] - old_bbox[0]
    ymin = new_bbox[1] - old_bbox[1]
    xmax = new_bbox[2] - old_bbox[2]
    ymax = new_bbox[3] - old_bbox[3]
    return xmin, ymin, xmax, ymax


def adjust_position_of_the_boxes(xmindiff, ymindiff, xmaxdiff, ymaxdiff, boxes):
    adjusted_boxes = []
    adjusted_box = []
    for b in boxes:
        xmin, ymin, xmax, ymax = b

        xmin -= xmindiff
        ymin -= ymindiff
        xmax -= xmaxdiff
        ymax -= ymaxdiff

        adjusted_box = [xmin, ymin, xmax, ymax]
        adjusted_boxes.append(adjusted_box)
    return adjusted_boxes


def get_center_of_box(box):
    xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return center_x, center_y


def calculate_new_position(box, center_x, center_y):
    xmin, ymin, xmax, ymax = box
    # Calculate the width and height of the original bounding box
    width1 = xmax - xmin
    height1 = ymax - ymin

    # Calculate the new top-left and bottom-right coordinates of the original bounding box
    xmin1_new = int(center_x - width1 / 2)
    ymin1_new = int(center_y - height1 / 2)
    xmax1_new = xmin1_new + width1
    ymax1_new = ymin1_new + height1

    return xmin1_new, ymin1_new, xmax1_new, ymax1_new


def get_difference_center_of_boxes(predict_center_x, predict_center_y, center_x, center_y):
    p = predict_center_x - center_x
    c = predict_center_y - center_y
    return p, c


def resize_image(image_path, image_width, image_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (image_width, image_height))
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image


def get_image_size(image):
    height = image.shape[0]
    width = image.shape[1]
    return width, height


def resize(input_path, output_path, width, height):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(output_path, resized_image)


def get_width_height_shape(image_path):
    image = cv2.imread(image_path)
    if image.shape[0] > YOLO_HEIGHT and image.shape[1] > YOLO_WIDTH:
        # height_ratio = YOLO_HEIGHT / image.shape[0]
        # width_ratio = YOLO_WIDTH / image.shape[1]
        height_ratio = image.shape[0] / YOLO_HEIGHT
        width_ratio = image.shape[1] / YOLO_WIDTH
    else:
        height_ratio = YOLO_HEIGHT / image.shape[0]
        width_ratio = YOLO_WIDTH / image.shape[1]
    return width_ratio, height_ratio

def scale_bounding_one_box(bbox, width_ratio, height_ratio):

    x_min = np.round(bbox[0] * width_ratio, 2)
    y_min = np.round(bbox[1] * height_ratio, 2)
    x_max = np.round(bbox[2] * width_ratio, 2)
    y_max = np.round(bbox[3] * height_ratio, 2)

    return [x_min, y_min, x_max, y_max]

def scale_bounding_box(bounding_boxes, width_ratio, height_ratio):
    scalde_bounding_boxes = []

    for box in bounding_boxes:
        x_min = np.round(box[0] * width_ratio, 2)
        y_min = np.round(box[1] * height_ratio, 2)
        x_max = np.round(box[2] * width_ratio, 2)
        y_max = np.round(box[3] * height_ratio, 2)
        scalde_bounding_boxes.append([x_min, y_min, x_max, y_max])
    return scalde_bounding_boxes

def scale_up(ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links, ratios):

    for i, box, cls in zip(range(len(ausbildung_cut_links[1])), ausbildung_cut_links[0], ausbildung_cut_links[1]):
        ausbildung_cut_links[0][i] = scale_bounding_one_box(box,ratios[0], ratios[1])

    for i, box, cls in zip(range(len(person_cut_links[1])), person_cut_links[0], person_cut_links[1]):
        person_cut_links[0][i] = scale_bounding_one_box(box,ratios[0], ratios[1])

    for i, box, cls in zip(range(len(wohnsitz_cut_links[1])), wohnsitz_cut_links[0], wohnsitz_cut_links[1]):
        wohnsitz_cut_links[0][i] = scale_bounding_one_box(box,ratios[0], ratios[1])

    for i, box, cls in zip(range(len(wwa_cut_links[1])), wwa_cut_links[0], wwa_cut_links[1]):
        wwa_cut_links[0][i] = scale_bounding_one_box(box,ratios[0], ratios[1])


    return ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links

def create_directories(output_path_images, output_path_annotations, new_width, new_height):
    output_path_images = output_path_images + '/' + str(new_height) + 'x' + str(new_width)
    output_path_annotations = output_path_annotations + '/' + str(new_height) + 'x' + str(new_width)
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_annotations, exist_ok=True)
