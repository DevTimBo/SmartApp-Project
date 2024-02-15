# Authors: Tim Harmling and Alexej Kravtschenko
# Simple config for the project which uses the coresponding file "configs.json"

import json

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def get_pipeline_parameter(self):
        return self.config["pipeline"]
    def get_model_parameter(self):
        return self.config["model"]

    def get_directory_parameter(self):
        return self.config["directory"]
    def get_training_parameter(self):
        return self.config["training"]
