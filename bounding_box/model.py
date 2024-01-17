import ressize
import keras_cv
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing import image as keras_image

from ressize import resize_image, get_width_height_shape, scale_bounding_box, calculate_bbox_scale_factor, scale_box, \
    resize_imaged_without_expand_dim, cut_links_bbox, cut_top_bbox, add_bottom_bbox, add_rechts_bbox, add_links_bbox, \
    get_position_difference_between_boxes, adjust_position_of_the_boxes, get_center_of_box, calculate_new_position, \
    get_difference_center_of_boxes

from config import LEARNING_RATE, GLOBAL_CLIPNORM, NUM_CLASSES_ALL, SUB_BBOX_DETECTOR_MODEL, BBOX_PATH, \
    MAIN_BBOX_DETECTOR_MODEL, class_ids, main_class_ids, sub_class_ids


def get_max_confidence_index(confidence):
    return np.argmax(confidence)


def get_ausbildung_index(classes):
    for i, cls in zip(range(len(classes)), classes):
        if cls == 0:
            return i


def get_org_ms_boxes_for_pred(best_predicted_class, org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung,
                              org_ms_boxes_wwa):
    if best_predicted_class == 0:
        pred_box = org_ms_boxes_ausbildung
        class_name = "Ausbildung"
    elif best_predicted_class == 1:
        pred_box = org_ms_boxes_person
        class_name = "Person"
    elif best_predicted_class == 2:
        pred_box = org_ms_boxes_wohnsitz
        class_name = "Wohnsitz"
    elif best_predicted_class == 3:
        pred_box = org_ms_boxes_wwa
        class_name = "Wohnsitz_waehrend_Ausbildung"
    else:
        pass
    return pred_box, class_name


def get_templated_data(boxes, confidence, classes, org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung,
                       org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids):
    # best_confidence_index = get_ausbildung_index(classes[0])
    # best_predicted_class = classes[0][best_confidence_index]
    # best_predicted_box = boxes[0][best_confidence_index]

    best_confidence_index = get_max_confidence_index(confidence[0])
    best_predicted_class = classes[0][best_confidence_index]
    best_predicted_box = boxes[0][best_confidence_index]

    org_ms_box, class_name = get_org_ms_boxes_for_pred(best_predicted_class, org_ms_boxes_person, org_ms_boxes_wohnsitz,
                                                       org_ms_boxes_ausbildung, org_ms_boxes_wwa)

    best_predicted = [best_predicted_box, best_predicted_class, best_confidence_index, class_name]

    scale_factor_weight, scale_factor_height = calculate_bbox_scale_factor(org_ms_box, best_predicted_box)
    scale_factor = [scale_factor_weight, scale_factor_height]
    template_resized_boxes = scale_bounding_box(org_ms_box, scale_factor_weight, scale_factor_height)

    xmindiff, ymindiff, xmaxdiff, ymaxdiff = get_position_difference_between_boxes(
        template_resized_boxes[(len(template_resized_boxes) - 1)], best_predicted_box)
    coordinate_difference = [xmindiff, ymindiff, xmaxdiff, ymaxdiff]

    adjust_position_ausbildung, adjust_position_person, adjust_position_wohnsitz, adjust_position_wwa = make_template_for_non_predicted_boxes(
        scale_factor, coordinate_difference, org_ms_boxes_ausbildung,
        org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_wwa)

    return [adjust_position_ausbildung, ausbildung_class_ids], [adjust_position_person, person_class_ids], [
        adjust_position_wohnsitz, wohnsitz_class_ids], [adjust_position_wwa, wwa_class_ids], best_predicted


def make_template_for_non_predicted_boxes(scale_factor, coordinate_difference, org_ms_boxes_ausbildung,
                                          org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_wwa):
    template_resized_ausbildung = scale_bounding_box(org_ms_boxes_ausbildung, scale_factor[0], scale_factor[1])
    adjust_position_ausbildung = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                              coordinate_difference[2], coordinate_difference[3],
                                                              template_resized_ausbildung)

    template_resized_person = scale_bounding_box(org_ms_boxes_person, scale_factor[0], scale_factor[1])
    adjust_position_person = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                          coordinate_difference[2], coordinate_difference[3],
                                                          template_resized_person)

    template_resized_wohnsitz = scale_bounding_box(org_ms_boxes_wohnsitz, scale_factor[0], scale_factor[1])
    adjust_position_wohnsitz = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                            coordinate_difference[2], coordinate_difference[3],
                                                            template_resized_wohnsitz)

    template_resized_wwa = scale_bounding_box(org_ms_boxes_wwa, scale_factor[0], scale_factor[1])
    adjust_position_wwa = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                       coordinate_difference[2], coordinate_difference[3],
                                                       template_resized_wwa)

    return adjust_position_ausbildung, adjust_position_person, adjust_position_wohnsitz, adjust_position_wwa


def predict_image(image, model):
    ratios = get_width_height_shape(image)
    resized_image = resize_image(image)
    predictions = model.predict(resized_image)
    boxes = predictions['boxes']
    boxes = scale_bounding_box(boxes, ratios[0], ratios[1])
    confidence = predictions['confidence']
    classes = predictions['classes']

    return boxes, confidence, classes


def define_model(num_classes):
    model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=define_backbone("yolo_v8_xs_backbone_coco"),
        fpn_depth=1,
    )
    return model


def compile_model(model):
    model.compile(
        optimizer=define_optimizer(),
        classification_loss="binary_crossentropy",
        box_loss="ciou"
    )


def load_weight_model(model_path):
    base_model = define_model(43)  # (len(get_class_mapping(model_path)[0]))
    compile_model(base_model)
    base_model.load_weights(model_path)
    return base_model


def define_optimizer():
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    return optimizer


def define_backbone(pretrained_model):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        pretrained_model,
        load_weights=True
    )
    return backbone


def get_class_mapping(model_path):
    if MAIN_BBOX_DETECTOR_MODEL in model_path:
        main_class_mapping = dict(zip(range(len(main_class_ids)), main_class_ids))
        return main_class_mapping
    if SUB_BBOX_DETECTOR_MODEL in model_path:
        sub_class_mapping = dict(zip(range(len(sub_class_ids)), sub_class_ids))
        return sub_class_mapping
    else:
        class_mapping = dict(zip(range(len(class_ids)), class_ids))
        return class_mapping


def extract_boxes(predictions_on_image):
    best_bboxes = {}
    class_id = []
    bbox = []
    confidence = []

    for i in range(0, predictions_on_image['num_detections'][0]):
        class_id.append(predictions_on_image['classes'][0][i])
        bbox.append(predictions_on_image['boxes'][0][i])
        confidence.append(predictions_on_image['confidence'][0][i])

    for i in range(len(class_id)):
        current_class = class_id[i]
        current_confidence = confidence[i]
        current_box = bbox[i]

        if current_class in best_bboxes and np.all(current_box > best_bboxes[current_class]):
            best_bboxes[current_class] = current_box
        elif current_class not in best_bboxes:
            best_bboxes[current_class] = current_box
    return best_bboxes


def get_image_as_array(image_path):
    image = cv2.imread(image_path)
    image = np.expand_dims(image, axis=0)
    return image


def non_maximum_supression(boxes, confidence, classes):
    selected_indices = tf.image.non_max_suppression(
        boxes, confidence, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

    return selected_boxes


def show_image(image, boxes, confidence, classes):
    image = cv2.imread(image)

    # image_with_boxes = np.copy(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, conf, cls in zip(boxes[0], confidence[0], classes[0]):
        if conf > 0.1:
            xmin, ymin, xmax, ymax = box
            label = f"Class {cls} ({conf:.2f})"
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                     facecolor='none', label=label)
            ax.add_patch(rect)

    plt.legend()
    plt.show()
