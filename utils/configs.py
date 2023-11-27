import json

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def get_model_params(self):
        return self.config.get("model_params", {})

    def get_training_params(self):
        return self.config.get("training_params", {})
    
    def print_model_params(self):
        model_params = self.get_model_params()
        print("Model Parameters:")
        for key, value in model_params.items():
            print(f"{key}: {value}") 
    
    def get_image_params(self):
        return self.config.get("image_params", {})
    
    def print_image_params(self):
        image_params = self.get_image_params()
        print("Image Parameters:")
        for key, value in image_params.items():
            print(f"{key}: {value}")
    
    def get_path_params(self):
        return self.config.get("path_params", {})
    
    def print_path_params(self):
        path_params = self.get_path_params()
        print("Path Parameters:")
        for key, value in path_params.items():
            print(f"{key}: {value}")  

    def print_training_params(self):
        training_params = self.get_training_params()
        print("Training Parameters:")
        for key, value in training_params.items():
            print(f"{key}: {value}")