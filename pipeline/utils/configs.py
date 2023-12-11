import json


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def get_handwriting_parameter(self):
        return self.config["handwriting"]

    def get_bounding_box_parameter(self):
        return self.config["bounding_box"]
