import load_data
from utils.configs import ModelConfig

config_path = "./utils/configs.json"
model_config = ModelConfig(config_path)

# Zugriff auf Modellparameter
model_config.print_model_params()
model_config.print_training_params()

model_params = model_config.get_model_params()
input_shape = model_params.get("input_shape")

print("Input:{}".format(input_shape))


load_data.read_data()
load_data.split_data()
load_data.create_train_and_test_data()
load_data.get_vocabulary_length_and_clean_labels()
load_data.clean_test_labels()

X_train_img_paths, y_train_labels = load_data.train_img_paths, load_data.train_labels_cleaned
X_test_img_paths, y_test_labels = load_data.test_img_paths, load_data.test_labels_cleaned

print("\n",X_test_img_paths[0:10], y_train_labels[0:10])


