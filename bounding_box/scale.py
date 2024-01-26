import cv2

def scale_image(input_path, output_path, scaling_height, scaling_width):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, (scaling_width, scaling_height))
    cv2.imwrite(output_path, resized_image)

def scale_box(image_path, bounding_box, scaling_height, scaling_width):
    image = cv2.imread(image_path)
    height_ratio = scaling_height / image.shape[0]
    width_ratio = scaling_width / image.shape[1]
    x_min = (bounding_box[0] * width_ratio)
    y_min = (bounding_box[1] * height_ratio)
    x_max = (bounding_box[2] * width_ratio)
    y_max = (bounding_box[3] * height_ratio)
    return [x_min, y_min, x_max, y_max]

def scale_up(image_path, ausbildung_cut, person_cut, wohnsitz_cut, wwa_cut, scaling_height, scaling_width):
    ausbildung_boxes = []
    person_boxes = []
    wohnsitz_boxes = []
    wwa_boxes = []
    for box in (ausbildung_cut[0]):
        ausbildung_boxes.append(scale_box(image_path, box, scaling_height, scaling_width))
        ausbildung_cut[0] = ausbildung_boxes

    for box in (person_cut[0]):
        person_boxes.append(scale_box(image_path, box, scaling_height, scaling_width))
        person_cut[0] = person_boxes

    for box in (wohnsitz_cut[0]):
        wohnsitz_boxes.append(scale_box(image_path, box, scaling_height, scaling_width))
        wohnsitz_cut[0] = wohnsitz_boxes

    for box in (wwa_cut[0]):
        wwa_boxes.append(scale_box(image_path, box, scaling_height, scaling_width))
        wwa_cut[0] = wwa_boxes

    return ausbildung_cut, person_cut, wohnsitz_cut, wwa_cut

