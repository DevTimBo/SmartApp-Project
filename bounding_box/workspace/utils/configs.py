import json


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def get_path_parameter(self):
        return self.config["PATH"]

    def get_class_info_parameter(self):
        return self.config["CLASS_INFO"]

    def get_class_ids_parameter(self):
        return self.config["CLASS_IDS"]

    def get_basename_parameter(self):
        return self.config["BASENAME"]

    def get_model_parameter(self):
        return self.config["MODEL"]

    def get_training_parameter(self):
        return self.config["TRAINING"]

    def get_WORKSPACE_AND_TEMPlATING_PATH(self, last_part):
        if last_part == "SmartApp-Project":  # path for pipeline
            TEMPlATING_ANNOTATION_PATH = "bounding_box/workspace/templating_data/Annotations"
            WORKSPACE_PATH = 'bounding_box/workspace/'
        elif last_part == "handwriting":
            WORKSPACE_PATH = '../bounding_box/workspace'
            TEMPlATING_ANNOTATION_PATH = "../bounding_box/workspace/templating_data/Annotations"
        else:
            WORKSPACE_PATH = 'workspace'
            TEMPlATING_ANNOTATION_PATH = "workspace/templating_data/Annotations"

        return WORKSPACE_PATH, TEMPlATING_ANNOTATION_PATH
