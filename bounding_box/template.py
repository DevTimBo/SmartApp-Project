import os
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from config import class_ids, main_class_ids, sub_class_ids, TEMPlATING_ANNOTATION_PATH

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
