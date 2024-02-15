#Autoren Tristan und Alireza
import os
cwd = os.getcwd()
last_part = os.path.basename(cwd)

if last_part == "SmartApp-Project": # path for pipeline
    TEMPlATING_ANNOTATION_PATH = "bounding_box/workspace/templating_data/Annotations"
    WORKSPACE_PATH = 'bounding_box/workspace/'
elif last_part == "handwriting":
    WORKSPACE_PATH = '../bounding_box/workspace'
    TEMPlATING_ANNOTATION_PATH = "../bounding_box/workspace/templating_data/Annotations"
else:
    WORKSPACE_PATH = 'workspace'
    TEMPlATING_ANNOTATION_PATH = "workspace/templating_data/Annotations"

ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
BBOX_PATH = MODEL_PATH + '/bbox'
MAIN_BBOX_DETECTOR_MODEL = MODEL_PATH + '/main_bbox_detector_model.h5'
SUB_BBOX_DETECTOR_MODEL = MODEL_PATH + '/sub_bbox_detector_model.h5'

# YOLO_HEIGHT = 640
# YOLO_WIDTH = 640

YOLO_HEIGHT = 1536
YOLO_WIDTH = 768

SPLIT_RATIO = 0.2
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCH = 50
GLOBAL_CLIPNORM = 10.0
NUM_CLASSES_ALL = 42
NUM_CLASSES_MAIN = 3
NUM_CLASSES_SUB = 39

class_ids = [
    "Ausbildung_Klasse",
    "Ausbildung_Antrag_gestellt_ja",
    "Ausbildung_Antrag_gestellt_nein",
    "Ausbildung_Amt",
    "Ausbildung_Foerderungsnummer",
    "Ausbilung_Abschluss",
    "Ausbildung_Vollzeit",
    "Ausbildung_Teilzeit",
    "Ausbildung_Staette",
    "Person_Geburtsort",
    "Person_maennlich",
    "Person_Geburtsdatum",
    "Person_weiblich",
    "Person_divers",
    "Person_Name",
    "Person_Familienstand",
    "Person_Vorname",
    "Person_Geburtsname",
    "Person_Familienstand_seit",
    "Person_Stattsangehörigkeit_eigene",
    "Person_Stattsangehörigkeit_Ehegatte",
    "Person_Kinder",
    "Wohnsitz_Strasse",
    "Wohnsitz_Land",
    "Wohnsitz_Postleitzahl",
    "Wohnsitz_Hausnummer",
    "Wohnsitz_Adresszusatz",
    "Wohnsitz_Ort",
    "Wohnsitz_waehrend_Ausbildung_Strasse",
    "Wohnsitz_waehrend_Ausbildung_Hausnummer",
    "Wohnsitz_waehrend_Ausbildung_Land",
    "Wohnsitz_waehrend_Ausbildung_Ort",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein",
    "Wohnsitz_waehrend_Ausbildung_Adresszusatz",
    "Wohnsitz_waehrend_Ausbildung_Postleitzahl",
    "Wohnsitz_waehrend_Ausbildung_elternmiete",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja",
    "Wohnsitz_waehrend_Ausbildung_elternmiete_nein",
    "Ausbildung",
    "Person",
    "Wohnsitz",
    "Wohnsitz_waehrend_Ausbildung", 
]
sub_class_ids = [
    "Ausbildung_Klasse",
    "Ausbildung_Antrag_gestellt_ja",
    "Ausbildung_Antrag_gestellt_nein",
    "Ausbildung_Amt",
    "Ausbildung_Foerderungsnummer",
    "Ausbilung_Abschluss",
    "Ausbildung_Vollzeit",
    "Ausbildung_Teilzeit",
    "Ausbildung_Staette",
    "Person_Geburtsort",
    "Person_maennlich",
    "Person_Geburtsdatum",
    "Person_weiblich",
    "Person_divers",
    "Person_Name",
    "Person_Familienstand",
    "Person_Vorname",
    "Person_Geburtsname",
    "Person_Familienstand_seit",
    "Person_Stattsangehörigkeit_eigene",
    "Person_Stattsangehörigkeit_Ehegatte",
    "Person_Kinder",
    "Wohnsitz_Strasse",
    "Wohnsitz_Land",
    "Wohnsitz_Postleitzahl",
    "Wohnsitz_Hausnummer",
    "Wohnsitz_Adresszusatz",
    "Wohnsitz_Ort",
    "Wohnsitz_waehrend_Ausbildung_Strasse",
    "Wohnsitz_waehrend_Ausbildung_Hausnummer",
    "Wohnsitz_waehrend_Ausbildung_Land",
    "Wohnsitz_waehrend_Ausbildung_Ort",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein",
    "Wohnsitz_waehrend_Ausbildung_Adresszusatz",
    "Wohnsitz_waehrend_Ausbildung_Postleitzahl",
    "Wohnsitz_waehrend_Ausbildung_elternmiete",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja",
    "Wohnsitz_waehrend_Ausbildung_elternmiete_nein"

]
main_class_ids = [
    "Ausbildung",
    "Person",
    "Wohnsitz",
    "Wohnsitz_waehrend_Ausbildung",
]
