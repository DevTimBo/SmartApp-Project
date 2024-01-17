import os
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from bounding_box.config import class_ids, main_class_ids, sub_class_ids, TEMPlATING_ANNOTATION_PATH


def get_mapping_classes():
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    main_class_mapping = dict(zip(range(len(main_class_ids)), main_class_ids))
    sub_class_mapping = dict(zip(range(len(sub_class_ids)), sub_class_ids))
    return class_mapping, main_class_mapping, sub_class_mapping


def get_xml_files(path):
    return sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".xml")
        ]
    )


def map_class_id(classes, class_mapping):
    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return class_ids


def create_box(bbox):
    xmin = float(bbox.find("xmin").text)
    ymin = float(bbox.find("ymin").text)
    xmax = float(bbox.find("xmax").text)
    ymax = float(bbox.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


def is_bbox1_inside_bbox2(bbox1, bbox2):
    bbox2 = bbox2[0]
    x1_in_range = float(bbox2[0]) <= float(bbox1[0]) <= float(bbox1[2]) <= float(bbox2[2])
    y1_in_range = float(bbox2[1]) <= float(bbox1[1]) <= float(bbox1[3]) <= float(bbox2[3])

    return x1_in_range and y1_in_range


def get_main_box_data(xml_file):
    main_boxes_person = []
    main_boxes_wohnsitz = []
    main_boxes_ausbildung = []
    main_boxes_wwa = []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        cls = obj.find("name").text
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

    return main_boxes_person, main_boxes_wohnsitz, main_boxes_ausbildung, main_boxes_wwa


def get_for_main_bbox_sub_bboxes(xml_file, main_boxes_person, main_boxes_ausbildung, main_boxes_wohnsitz,
                                 main_boxes_wwa, class_mapping):
    main_sub_boxes_person = []
    main_sub_boxes_wohnsitz = []
    main_sub_boxes_ausbildung = []
    main_sub_boxes_wwa = []
    main_sub_classes_person = []
    main_sub_classes_wohnsitz = []
    main_sub_classes_ausbildung = []
    main_sub_classes_wwa = []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    widthElement = root.find("size/width")
    heightElement = root.find("size/height")
    for obj in root.iter("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")

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

    person_class_ids = map_class_id(main_sub_classes_person, class_mapping)
    ausbildung_class_ids = map_class_id(main_sub_classes_ausbildung, class_mapping)
    wohnsitz_class_ids = map_class_id(main_sub_classes_wohnsitz, class_mapping)
    wwa_class_ids = map_class_id(main_sub_classes_wwa, class_mapping)

    return main_sub_boxes_person, main_sub_boxes_wohnsitz, main_sub_boxes_ausbildung, main_sub_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, int(
        widthElement.text), int(heightElement.text)


def build_templating_data():
    class_mapping, main_class_mapping, sub_class_mapping = get_mapping_classes()

    xml_files = get_xml_files(TEMPlATING_ANNOTATION_PATH)

    for xml_file in tqdm(xml_files):
        main_boxes_person, main_boxes_wohnsitz, main_boxes_ausbildung, main_boxes_wwa = get_main_box_data(
            xml_file)

    for xml_file in tqdm(xml_files):
        org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag = get_for_main_bbox_sub_bboxes(
            xml_file, main_boxes_person, main_boxes_ausbildung, main_boxes_wohnsitz, main_boxes_wwa, class_mapping)
    return org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag
