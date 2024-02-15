#5010890
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def read_annotations_from_xml(xml_path, image_name):
    """
    Read annotations for a specific image from an XML file.

    :param xml_path: Path to the XML file.
    :param image_name: Name of the image to get annotations for.
    :return: List of annotations, where each annotation is a dictionary with 'xtl', 'ytl', 'xbr', 'ybr'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    # Find the image element and its bounding boxes
    for image in root.findall('image'):
        if image.get('name') == image_name:
            for box in image.findall('box'):
                annotations.append({
                    'xtl': float(box.get('xtl')),
                    'ytl': float(box.get('ytl')),
                    'xbr': float(box.get('xbr')),
                    'ybr': float(box.get('ybr'))
                })
    return annotations

def visualize_annotations(image_path, annotations):
    """
    Visualize annotations on an image.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        bbox = (ann['xtl'], ann['ytl'], ann['xbr'], ann['ybr'])
        draw.rectangle(bbox, outline='red', width=2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Example usage
xml_path = 'small/annotations.xml'  
image_name = 'image_0001.jpg'    
image_path = 'small/image_0001.jpg'   

annotations = read_annotations_from_xml(xml_path, image_name)
visualize_annotations(image_path, annotations)
