#Autor Alireza
import os
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from bounding_box.config import class_ids, main_class_ids, sub_class_ids, TEMPlATING_ANNOTATION_PATH


def get_mapping_classes():
    # Create mappings for class IDs.
    # Create a mapping dictionary for all class IDs
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    # Create a mapping dictionary for main class IDs
    main_class_mapping = dict(zip(range(len(main_class_ids)), main_class_ids))
    # Create a mapping dictionary for sub class IDs
    sub_class_mapping = dict(zip(range(len(sub_class_ids)), sub_class_ids))
    return class_mapping, main_class_mapping, sub_class_mapping


def get_xml_files(path):
    # Get a sorted list of XML files in the specified directory.
    return sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".xml")
        ]
    )


def map_class_id(classes, class_mapping):
    # Map class names to class IDs based on a given mapping dictionary.
    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return class_ids


def create_box(bbox):
    # # Extract xmin, ymin, xmax, ymax coordinates from the XML element
    xmin = float(bbox.find("xmin").text)
    ymin = float(bbox.find("ymin").text)
    xmax = float(bbox.find("xmax").text)
    ymax = float(bbox.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


def is_bbox1_inside_bbox2(bbox1, bbox2):
    #  Check if bbox1 is completely inside bbox2.

    # Extract coordinates of bbox2
    bbox2 = bbox2[0]
    # Check if x-coordinates of bbox1 are within the x-range of bbox2
    x1_in_range = float(bbox2[0]) <= float(bbox1[0]) <= float(bbox1[2]) <= float(bbox2[2])
    # Check if y-coordinates of bbox1 are within the y-range of bbox2
    y1_in_range = float(bbox2[1]) <= float(bbox1[1]) <= float(bbox1[3]) <= float(bbox2[3])

    # Return True if both x and y coordinates are within the respective ranges, False otherwise
    return x1_in_range and y1_in_range


def get_main_box_data(xml_file):
    # Extract main boxes data from XML file.
    # Initialize lists to store main boxes for each main class
    main_boxes_person = []
    main_boxes_wohnsitz = []
    main_boxes_ausbildung = []
    main_boxes_wwa = []
    # Parse the XML tree
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Iterate through each object in the XML tree
    for obj in root.iter("object"):
        # Extract class label
        cls = obj.find("name").text
        # Append the bounding box to the appropriate list based on class label
        if cls == 'Person':
            main_boxes_person.append(create_box(obj.find("bndbox")))
        elif cls == 'Wohnsitz':
            main_boxes_wohnsitz.append(create_box(obj.find("bndbox")))
        elif cls == 'Ausbildung':
            main_boxes_ausbildung.append(create_box(obj.find("bndbox")))
        elif cls == 'Wohnsitz_waehrend_Ausbildung':
            main_boxes_wwa.append(create_box(obj.find("bndbox")))
        else:
            pass
    # Return lists of main boxes for each main class
    return main_boxes_person, main_boxes_wohnsitz, main_boxes_ausbildung, main_boxes_wwa


def get_for_main_bbox_sub_bboxes(xml_file, main_boxes_person, main_boxes_ausbildung, main_boxes_wohnsitz,
                                 main_boxes_wwa, class_mapping):
    #  Extract sub boxes and their corresponding class IDs for each main class from the XML file
    #  Initialize lists to store sub boxes and their class IDs for each main class
    main_sub_boxes_person = []
    main_sub_boxes_wohnsitz = []
    main_sub_boxes_ausbildung = []
    main_sub_boxes_wwa = []
    main_sub_classes_person = []
    main_sub_classes_wohnsitz = []
    main_sub_classes_ausbildung = []
    main_sub_classes_wwa = []
    # Parse the XML tree
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find width and height elements in the XML tree
    widthElement = root.find("size/width")
    heightElement = root.find("size/height")
    # Iterate through each object in the XML tree
    for obj in root.iter("object"):
        # Extract class label and bounding box
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        # Check if the bounding box is inside the main box for each main class
        if is_bbox1_inside_bbox2(create_box(bbox), main_boxes_person):
            main_sub_boxes_person.append(create_box(bbox))
            main_sub_classes_person.append(cls)
        elif is_bbox1_inside_bbox2(create_box(bbox), main_boxes_ausbildung):
            main_sub_boxes_ausbildung.append(create_box(bbox))
            main_sub_classes_ausbildung.append(cls)
        elif is_bbox1_inside_bbox2(create_box(bbox), main_boxes_wohnsitz):
            main_sub_boxes_wohnsitz.append(create_box(bbox))
            main_sub_classes_wohnsitz.append(cls)
        elif is_bbox1_inside_bbox2(create_box(bbox), main_boxes_wwa):
            main_sub_boxes_wwa.append(create_box(bbox))
            main_sub_classes_wwa.append(cls)
        else:
            pass
    # Map class names to class IDs for each main class
    person_class_ids = map_class_id(main_sub_classes_person, class_mapping)
    ausbildung_class_ids = map_class_id(main_sub_classes_ausbildung, class_mapping)
    wohnsitz_class_ids = map_class_id(main_sub_classes_wohnsitz, class_mapping)
    wwa_class_ids = map_class_id(main_sub_classes_wwa, class_mapping)
    # Return lists of sub boxes and their class IDs for each main class, along with image dimensions
    return main_sub_boxes_person, main_sub_boxes_wohnsitz, main_sub_boxes_ausbildung, main_sub_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, int(
        widthElement.text), int(heightElement.text)


def build_templating_data():
    # Build templating data from XML annotations.
    # Get class mapping dictionaries
    class_mapping, main_class_mapping, sub_class_mapping = get_mapping_classes()
    # Get list of XML files containing annotations
    xml_files = get_xml_files(TEMPlATING_ANNOTATION_PATH)

    # Iterate over each XML file
    for xml_file in tqdm(xml_files):
        # Extract main boxes data
        main_boxes_person, main_boxes_wohnsitz, main_boxes_ausbildung, main_boxes_wwa = get_main_box_data(
            xml_file)
    # Iterate over each XML file again
    for xml_file in tqdm(xml_files):
        # Get sub boxes and class IDs for each main class
        org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag = get_for_main_bbox_sub_bboxes(
            xml_file, main_boxes_person, main_boxes_ausbildung, main_boxes_wohnsitz, main_boxes_wwa, class_mapping)
        # Return original main sub boxes, class IDs, and image dimensions
    return org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag
