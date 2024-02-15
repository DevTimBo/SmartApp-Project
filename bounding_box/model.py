#Autoren Tristan und Alireza
import keras_cv
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bounding_box.ressize import resize_image, get_width_height_shape, scale_bounding_box, calculate_bbox_scale_factor, \
    cut_links_bbox, cut_top_bbox, add_bottom_bbox, add_rechts_bbox, add_links_bbox, \
    get_position_difference_between_boxes, adjust_position_of_the_boxes, cut_rechts_bbox, add_top_bbox

from bounding_box.config import LEARNING_RATE, GLOBAL_CLIPNORM, YOLO_WIDTH, YOLO_HEIGHT


def get_max_confidence_index(confidence):
    # Function to find the index of the maximum confidence value in an array.
    return np.argmax(confidence)


def get_org_ms_boxes_for_pred(best_predicted_class, org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung,
                              org_ms_boxes_wwa):
    # Function to get the original bounding boxes based on the best predicted class
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
    # get best prediction data
    best_confidence_index = get_max_confidence_index(confidence[0])
    best_predicted_class = classes[0][best_confidence_index]
    best_predicted_box = boxes[0][best_confidence_index]
    # get original main box based on best predicted bbox
    org_ms_box, class_name = get_org_ms_boxes_for_pred(best_predicted_class, org_ms_boxes_person, org_ms_boxes_wohnsitz,
                                                       org_ms_boxes_ausbildung, org_ms_boxes_wwa)
    # save data in an array
    best_predicted = [best_predicted_box, best_predicted_class, best_confidence_index, class_name]
    # find scale factor base on predicted bbox and original bbox
    scale_factor_weight, scale_factor_height = calculate_bbox_scale_factor(org_ms_box, best_predicted_box)
    # save scale data in an array
    scale_factor = [scale_factor_weight, scale_factor_height]
    # resize original main bbox base on scale factor
    template_resized_boxes = scale_bounding_box(org_ms_box, scale_factor_weight, scale_factor_height)
    # find coordiante difference between predicted and original bbox
    xmindiff, ymindiff, xmaxdiff, ymaxdiff = get_position_difference_between_boxes(
        template_resized_boxes[(len(template_resized_boxes) - 1)], best_predicted_box)
    # save coordinate in an array
    coordinate_difference = [xmindiff, ymindiff, xmaxdiff, ymaxdiff]
    # base on all information, build templating for all bboxes
    adjust_position_ausbildung, adjust_position_person, adjust_position_wohnsitz, adjust_position_wwa = make_template_for_non_predicted_boxes(
        scale_factor, coordinate_difference, org_ms_boxes_ausbildung,
        org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_wwa)

    return [adjust_position_ausbildung, ausbildung_class_ids], [adjust_position_person, person_class_ids], [
        adjust_position_wohnsitz, wohnsitz_class_ids], [adjust_position_wwa, wwa_class_ids], best_predicted


def make_template_for_non_predicted_boxes(scale_factor, coordinate_difference, org_ms_boxes_ausbildung,
                                          org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_wwa):
    template_resized_ausbildung = scale_bounding_box(org_ms_boxes_ausbildung, scale_factor[0], scale_factor[1])
    # adjust position of the ausbildung boxes
    adjust_position_ausbildung = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                              coordinate_difference[2], coordinate_difference[3],
                                                              template_resized_ausbildung)
    # resize person bboxes
    template_resized_person = scale_bounding_box(org_ms_boxes_person, scale_factor[0], scale_factor[1])
    # adjust position of the person boxes
    adjust_position_person = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                          coordinate_difference[2], coordinate_difference[3],
                                                          template_resized_person)
    # resize wohnsitz bboxes
    template_resized_wohnsitz = scale_bounding_box(org_ms_boxes_wohnsitz, scale_factor[0], scale_factor[1])
    # adjust position of the wohnsitz boxes
    adjust_position_wohnsitz = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                            coordinate_difference[2], coordinate_difference[3],
                                                            template_resized_wohnsitz)
    # resize wohnsitz-während-ausbildung (wwa) bboxes
    template_resized_wwa = scale_bounding_box(org_ms_boxes_wwa, scale_factor[0], scale_factor[1])
    # adjust position of the wwa boxes
    adjust_position_wwa = adjust_position_of_the_boxes(coordinate_difference[0], coordinate_difference[1],
                                                       coordinate_difference[2], coordinate_difference[3],
                                                       template_resized_wwa)

    return adjust_position_ausbildung, adjust_position_person, adjust_position_wohnsitz, adjust_position_wwa


def edit_sub_boxes_cut_links(ausbildung, person, wohnsitz, wwa):
    # function to adjust the sizes of the sub box so that only the handwritten text is visible
    # cut the box from the --> left side <-- for this purpose.
    # For some boxes, increase the size on the right, top, or bottom to include all possible handwritten text within the box.

    # ausbildung boxes [1, 6, 5, 4, 0, 8, 3, 38]
    for i, box, cls in zip(range(len(ausbildung[1])), ausbildung[0], ausbildung[1]):
        if cls == 8:  # Ausbildung_Staette
            ausbildung[0][i][0] = cut_links_bbox(0.67, box)

        elif cls == 0:  # Ausbildung_Klasse
            ausbildung[0][i][0] = cut_links_bbox(0.84, box)

        elif cls == 5:  # "Ausbilung_Abschluss"
            ausbildung[0][i][0] = cut_links_bbox(0.8, box)

        elif cls == 6:  # "Ausbildung_Vollzeit","]
            ausbildung[0][i][0] = cut_links_bbox(0.3, box)

        elif cls == 1:  # "Ausbildung_Antrag_gestellt_ja",
            ausbildung[0][i][0] = cut_links_bbox(0.3, box)

        elif cls == 3:  # "Ausbildung_Amt"
            ausbildung[0][i][0] = cut_links_bbox(0.3, box)


        elif cls == 4:  # "Ausbildung_Foerderungsnummer",
            ausbildung[0][i][1] = cut_top_bbox(0.45, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

    # person boxes [15, 21, 20, 19, 18, 11, 9, 17, 16, 14, 39]
    for i, box, cls in zip(range(len(person[1])), person[0], person[1]):
        if cls == 14:  # "Person_Name",
            person[0][i][0] = cut_links_bbox(0.8, box)
            person[0][i][2] = add_rechts_bbox(2, box)

        elif cls == 16:  # "Person_Vorname",
            person[0][i][0] = cut_links_bbox(0.8, box)
            person[0][i][2] = add_rechts_bbox(0.5, box)

        elif cls == 17:  # "Person_Geburtsname",
            person[0][i][0] = cut_links_bbox(0.7, box)
            person[0][i][2] = add_rechts_bbox(0.5, box)

        elif cls == 9:  # "Person_Geburtsort",
            person[0][i][0] = cut_links_bbox(0.8, box)
            person[0][i][2] = add_rechts_bbox(0.2, box)

        elif cls == 11:  # "Person_Geburtsdatum"
            person[0][i][1] = cut_top_bbox(0.33, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 15:  # "Person_Familienstand"
            person[0][i][1] = cut_top_bbox(0.33, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 18:  # "Person_Familienstand_seit",
            person[0][i][0] = cut_links_bbox(0.65, box)
            person[0][i][1] = cut_top_bbox(0.2, box)
            person[0][i][3] = add_bottom_bbox(0.2, box)
        elif cls == 19:  # "Person_Stattsangehörigkeit_eigene",
            person[0][i][0] = cut_links_bbox(0.4, box)
            person[0][i][2] = add_rechts_bbox(0.3, box)
        elif cls == 20:  # "Person_Stattsangehörigkeit_Ehegatte",
            # person[0][i][3] = add_bottom_bbox(0.2, box)
            person[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 21:  # "Person_Kinder",
            person[0][i][0] = cut_links_bbox(0.25, box)
            person[0][i][2] = add_rechts_bbox(0.1, box)
            person[0][i][1] = add_top_bbox(0.15, box)

    # wohnsitz boxes [26, 23, 27, 24, 25, 22, 40]
    for i, box, cls in zip(range(len(wohnsitz[1])), wohnsitz[0], wohnsitz[1]):
        if cls == 22:  # "Wohnsitz_Strasse",
            wohnsitz[0][i][0] = cut_links_bbox(0.9, box)
            # wohnsitz[0][i][2] = add_rechts_bbox(0.2, box)

        elif cls == 25:  # "Wohnsitz_Hausnummer",
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)
            wohnsitz[0][i][3] = add_bottom_bbox(0.5, box)

        elif cls == 26:  # "Wohnsitz_Adresszusatz",
            wohnsitz[0][i][0] = cut_links_bbox(0.68, box)

        elif cls == 23:  # "Wohnsitz_Land",
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 24:  # "Wohnsitz_Postleitzahl",
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)
            wohnsitz[0][i][3] = add_bottom_bbox(0.4, box)
        elif cls == 27:  # "Wohnsitz_Ort",
            wohnsitz[0][i][0] = cut_links_bbox(0.85, box)
            wohnsitz[0][i][2] = add_rechts_bbox(1, box)

    # wwa boxes [33, 30, 31, 34, 29, 28, 41]
    for i, box, cls in zip(range(len(wwa[1])), wwa[0], wwa[1]):
        if cls == 28:  # "Wohnsitz_waehrend_Ausbildung_Strasse",
            wwa[0][i][0] = cut_links_bbox(0.8, box)
            wwa[0][i][2] = cut_rechts_bbox(0.1, box)

        elif cls == 29:  # "Wohnsitz_waehrend_Ausbildung_Hausnummer",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.15, box)

        elif cls == 33:  # "Wohnsitz_waehrend_Ausbildung_Adresszusatz",
            wwa[0][i][0] = cut_links_bbox(0.68, box)

        elif cls == 30:  # "Wohnsitz_waehrend_Ausbildung_Land",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 34:  # "Wohnsitz_waehrend_Ausbildung_Postleitzahl",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 31:  # "Wohnsitz_waehrend_Ausbildung_ort",
            wwa[0][i][0] = cut_links_bbox(0.90, box)

    return ausbildung, person, wohnsitz, wwa


def edit_sub_boxes_cut_top(ausbildung, person, wohnsitz, wwa):
    # function to adjust the sizes of the sub box so that only the handwritten text is visible
    # cut the box from the --> top side <-- for this purpose.
    # For some boxes, increase the size on the right, top, or bottom to include all possible handwritten text within the box.

    # ausbildung boxes [1, 6, 5, 4, 0, 8, 3, 38]
    for i, box, cls in zip(range(len(ausbildung[1])), ausbildung[0], ausbildung[1]):
        print(cls)
        if cls == 8:  # Ausbildung_Staette
            ausbildung[0][i][1] = cut_top_bbox(0.3, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 0:  # Ausbildung_Klasse
            ausbildung[0][i][1] = cut_top_bbox(0.3, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 5:  # "Ausbilung_Abschluss"
            ausbildung[0][i][1] = cut_top_bbox(0.3, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 6:  # "Ausbildung_Vollzeit","]

            ausbildung[0][i][0] = cut_links_bbox(0.4, box)

        elif cls == 1:  # "Ausbildung_Antrag_gestellt_ja",

            ausbildung[0][i][0] = cut_links_bbox(0.4, box)

        elif cls == 3:  # "Ausbildung_Amt"
            ausbildung[0][i][1] = cut_top_bbox(0.3, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 4:  # "Ausbildung_Foerderungsnummer",
            ausbildung[0][i][1] = cut_top_bbox(0.3, box)
            ausbildung[0][i][3] = add_bottom_bbox(0.3, box)

    # person boxes [15, 21, 20, 19, 18, 11, 9, 17, 16, 14, 39]
    for i, box, cls in zip(range(len(person[1])), person[0], person[1]):
        if cls == 14:  # "Person_Name",
            person[0][i][2] = add_rechts_bbox(1.5, box)
            person[0][i][0] = add_links_bbox(0.04, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 16:  # "Person_Vorname",
            person[0][i][2] = add_rechts_bbox(0.3, box)
            person[0][i][0] = add_links_bbox(0.04, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 17:  # "Person_Geburtsname",
            person[0][i][2] = add_rechts_bbox(0.45, box)
            person[0][i][0] = add_links_bbox(0.04, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 9:  # "Person_Geburtsort",
            person[0][i][2] = add_rechts_bbox(0.3, box)
            person[0][i][0] = add_links_bbox(0.04, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 11:  # "Person_Geburtsdatum"
            person[0][i][1] = cut_top_bbox(0.3, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 15:  # "Person_Familienstand"
            person[0][i][1] = cut_top_bbox(0.3, box)
            person[0][i][3] = add_bottom_bbox(0.3, box)

        elif cls == 18:  # "Person_Familienstand_seit",
            person[0][i][3] = add_bottom_bbox(0.2, box)
            person[0][i][0] = cut_links_bbox(0.65, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 19:  # "Person_Stattsangehörigkeit_eigene",
            person[0][i][3] = add_bottom_bbox(0.2, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 20:  # "Person_Stattsangehörigkeit_Ehegatte",
            person[0][i][3] = add_bottom_bbox(0.2, box)
            person[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 21:  # "Person_Kinder",

            person[0][i][0] = cut_links_bbox(0.2, box)

    # wohnsitz boxes [26, 23, 27, 24, 25, 22, 40]

    for i, box, cls in zip(range(len(wohnsitz[1])), wohnsitz[0], wohnsitz[1]):
        if cls == 22:  # "Wohnsitz_Strasse",
            # wohnsitz[0][i][2] = add_rechts_bbox(0.1, box)
            wohnsitz[0][i][0] = add_links_bbox(0.04, box)
            wohnsitz[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 25:  # "Wohnsitz_Hausnummer",
            wohnsitz[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 26:  # "Wohnsitz_Adresszusatz",
            # wohnsitz[0][i][2] = add_rechts_bbox(0.08, box)
            wohnsitz[0][i][0] = add_links_bbox(0.02, box)
            wohnsitz[0][i][1] = cut_top_bbox(0.1, box)

        elif cls == 23:  # "Wohnsitz_Land",
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 24:  # "Wohnsitz_Postleitzahl",
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)

        elif cls == 27:  # "Wohnsitz_Ort",
            wohnsitz[0][i][2] = add_rechts_bbox(0.6, box)
            wohnsitz[0][i][0] = add_links_bbox(0.02, box)
            wohnsitz[0][i][1] = cut_top_bbox(0.3, box)

    # wwa boxes [33, 30, 31, 34, 29, 28, 41]
    for i, box, cls in zip(range(len(wwa[1])), wwa[0], wwa[1]):
        if cls == 28:  # "Wohnsitz_waehrend_Ausbildung_Strasse",
            # wwa[0][i][2] = add_rechts_bbox(0.2, box)
            wwa[0][i][0] = add_links_bbox(0.04, box)
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.1, box)

        elif cls == 29:  # "Wohnsitz_waehrend_Ausbildung_Hausnummer",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.3, box)

        elif cls == 33:  # "Wohnsitz_waehrend_Ausbildung_Adresszusatz",
            # wwa[0][i][2] = add_rechts_bbox(0.08, box)
            wwa[0][i][0] = add_links_bbox(0.02, box)
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.1, box)

        elif cls == 30:  # "Wohnsitz_waehrend_Ausbildung_Land",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.1, box)

        elif cls == 34:  # "Wohnsitz_waehrend_Ausbildung_Postleitzahl",
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.1, box)

        elif cls == 31:  # "Wohnsitz_waehrend_Ausbildung_ort",
            # wwa[0][i][2] = add_rechts_bbox(0.1, box)
            wwa[0][i][0] = add_links_bbox(0.04, box)
            wwa[0][i][3] = add_bottom_bbox(0.3, box)
            wwa[0][i][1] = cut_top_bbox(0.1, box)

    return ausbildung, person, wohnsitz, wwa


def edit_sub_boxes_cut_left_and_top(ausbildung, person, wohnsitz, wwa):
    # function to cut left and top
    # this function can use only for the old dataset
    ausbildung, person, wohnsitz, wwa = edit_sub_boxes_cut_links(ausbildung, person, wohnsitz, wwa)
    return edit_sub_boxes_cut_top(ausbildung, person, wohnsitz, wwa)


def edit_sub_boxes_cut_left_or_top(ausbildung, person, wohnsitz, wwa):
    # this function can use only for the old dataset
    # function to cut left or top
    # for some sub box ist better to cut left and for the others ist better to cut top

    # ausbildung boxes [38, 8, 0, 5, 6, 7, 1, 2, 3, 4]
    for i, box, cls in zip(range(len(ausbildung[1])), ausbildung[0], ausbildung[1]):
        if cls == 8:
            ausbildung[0][i][0] = cut_links_bbox(0.64, box)
        elif cls == 0:
            ausbildung[0][i][0] = cut_links_bbox(0.78, box)
        elif cls == 5:
            ausbildung[0][i][0] = cut_links_bbox(0.75, box)
        elif cls == 3:
            ausbildung[0][i][0] = cut_links_bbox(0.3, box)
        # new
        elif cls == 4:
            ausbildung[0][i][1] = cut_top_bbox(0.45, box)

    # person boxes [39, 14, 16, 17, 9, 12, 10, 13, 11, 15, 18, 19, 20, 21]
    for i, box, cls in zip(range(len(person[1])), person[0], person[1]):
        if cls == 14:
            person[0][i][0] = cut_links_bbox(0.88, box)
        elif cls == 16:
            person[0][i][0] = cut_links_bbox(0.70, box)
        elif cls == 17:
            person[0][i][0] = cut_links_bbox(0.70, box)
        elif cls == 9:
            person[0][i][0] = cut_links_bbox(0.70, box)
        elif cls == 19:
            person[0][i][0] = cut_links_bbox(0.45, box)
        # new
        elif cls == 11:
            person[0][i][1] = cut_top_bbox(0.3, box)
        elif cls == 15:
            person[0][i][1] = cut_top_bbox(0.3, box)
        elif cls == 18:
            person[0][i][1] = cut_top_bbox(0.3, box)
        elif cls == 20:
            person[0][i][1] = cut_top_bbox(0.3, box)
    # wohnsitz boxes [40, 22, 25, 26, 23, 24, 27]
    for i, box, cls in zip(range(len(wohnsitz[1])), wohnsitz[0], wohnsitz[1]):
        if cls == 22:
            wohnsitz[0][i][0] = cut_links_bbox(0.75, box)
        elif cls == 26:
            wohnsitz[0][i][0] = cut_links_bbox(0.68, box)
        elif cls == 27:
            wohnsitz[0][i][0] = cut_links_bbox(0.90, box)
        # new
        elif cls == 25:
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)
        elif cls == 23:
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)
        elif cls == 24:
            wohnsitz[0][i][1] = cut_top_bbox(0.2, box)

    # wwa boxes [41, 28, 29, 30, 31, 34, 35]
    for i, box, cls in zip(range(len(wwa[1])), wwa[0], wwa[1]):
        if cls == 28:
            wwa[0][i][0] = cut_links_bbox(0.75, box)
        elif cls == 33:
            wwa[0][i][0] = cut_links_bbox(0.68, box)
        elif cls == 31:
            wwa[0][i][0] = cut_links_bbox(0.90, box)
            # new
        elif cls == 29:
            wwa[0][i][1] = cut_top_bbox(0.3, box)
            wwa[0][i][3] = add_bottom_bbox(0.2, box)
        elif cls == 30:
            wwa[0][i][1] = cut_top_bbox(0.3, box)
            wwa[0][i][3] = add_bottom_bbox(0.2, box)
        elif cls == 34:
            wwa[0][i][1] = cut_top_bbox(0.3, box)
            wwa[0][i][3] = add_bottom_bbox(0.2, box)

    return ausbildung, person, wohnsitz, wwa


def plot_image(image, ausbildung, person, wohnsitz, wwa, best_predicted):
    # plot image with template

    # Read the image
    image = cv2.imread(image)
    # Create a plot
    fig, ax = plt.subplots(1)
    # Display the image on the plot
    ax.imshow(image)

    # Plot bounding boxes for class 'ausbildung'
    for b in ausbildung[0]:
        xmin, ymin, xmax, ymax = b
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    # Plot bounding boxes for class 'person'
    for b in person[0]:
        xmin, ymin, xmax, ymax = b
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot bounding boxes for class 'wohnsitz'
    for b in wohnsitz[0]:
        xmin, ymin, xmax, ymax = b
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot bounding boxes for class 'wwa'
    for b in wwa[0]:
        xmin, ymin, xmax, ymax = b
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    # Show legend
    plt.legend()
    # Show the plot
    plt.show()


def predict_image(image, model):
    # get shape info
    ratios = get_width_height_shape(image)
    # resize image base on training-image-size
    resized_image = resize_image(image, YOLO_WIDTH, YOLO_HEIGHT)
    # predict
    predictions = model.predict(resized_image)
    # save predicted data in local variable
    boxes = predictions['boxes']
    confidence = predictions['confidence']
    classes = predictions['classes']

    return boxes, confidence, classes, ratios


def define_model(num_classes):
    #  function creates a YOLO model using the YOLOV8Detector
    model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=define_backbone("yolo_v8_xs_backbone_coco"),
        fpn_depth=1,  # Specify the depth of the feature pyramid network
    )
    return model


def compile_model(model):
    # function compiles the given model using the defined optimizer, classification loss, and box loss
    model.compile(
        optimizer=define_optimizer(),
        classification_loss="binary_crossentropy",  # Specify binary cross-entropy loss for classification
        box_loss="ciou"  # Specify custom CIoU loss for bounding box regression
    )


def load_weight_model(model_path, number_of_classes):
    # Create a model based on the number of classes the original model was trained on
    base_model = define_model(number_of_classes)
    # compile the model
    compile_model(base_model)
    # load weights
    base_model.load_weights(model_path)
    return base_model


def define_optimizer():
    # function defines an Adam optimizer for the model with the specified learning rate and global clipnorm.
    # It then returns the optimizer.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    return optimizer


def define_backbone(pretrained_model):
    # Define a YOLO backbone using the specified pretrained model
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        pretrained_model,
        load_weights=True
    )
    return backbone


def get_image_as_array(image_path):
    # This function takes the path to an image as input, reads the image using OpenCV, and converts it to a NumPy array.
    image = cv2.imread(image_path)
    image = np.expand_dims(image, axis=0)
    return image


def show_image(image, boxes, confidence, classes):
    # function to plot image and all predicted bboxes, conficence and classes

    # Read the image using OpenCV
    image = cv2.imread(image)
    image = cv2.resize(image, (YOLO_WIDTH, YOLO_HEIGHT))
    # Create a plot to display the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Iterate through each detected object
    for box, conf, cls in zip(boxes[0], confidence[0], classes[0]):
        if conf > 0.1:
            xmin, ymin, xmax, ymax = box
            # Construct label with class and confidence
            label = f"Class {cls} ({conf:.2f})"
            # Draw bounding box rectangle
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                     facecolor='none', label=label)
            # Add bounding box rectangle to the plot
            ax.add_patch(rect)
    # Add legend to the plot
    plt.legend()
    # Show the plot
    plt.show()
