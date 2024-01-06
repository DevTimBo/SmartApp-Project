# Unsere Klassen
import handwriting.preprocess
import utils.configs as Config
# Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import StringLookup
from path import Path
import numpy as np
import argparse
import pathlib
import pickle
import json 
import os


config_path = "utils/configs.json"
config = Config.Config(config_path)

# Model Parameter
MODEL_SAVE = bool(config.get_model_parameter()["save"])
MODEL_NAME = config.get_model_parameter()["name"]
IMAGE_WIDTH = config.get_model_parameter()["width"] # default: 1024
IMAGE_HEIGHT = config.get_model_parameter()["height"] # default: 64

# Directory Parameter
MODEL_DIR_NAME = pathlib.Path(os.getcwd()).joinpath(config.get_directory_parameter()["model_dir"])
TEST_RESULT_DIR_NAME = pathlib.Path(os.getcwd()).joinpath(config.get_directory_parameter()["test_dir"])
DATA_BASE_PATH = config.get_directory_parameter()["data_base_path"]

# Training Parameter
SAVE_HISTORY = bool(config.get_training_parameter()["save_history"])
EPOCHS = config.get_training_parameter()["epochs"]
BATCH_SIZE = config.get_training_parameter()["batch_size"] # default: 32 - 48
TF_SEED = config.get_training_parameter()["tf_seed"] # default: 42
LEARNING_RATE = config.get_training_parameter()["learning_rate"]
PATIENCE = config.get_training_parameter()["patience"] # default: 3

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='Choose which mode you\'d like to run', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch', 'none'], default='none')
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='./handwriting/data/test.png')
    parser.add_argument('--get_config', help='Get the current configuration', action='store_true')    
    
    #parser.add_argument('--set_config', help='Set a new configuration file path', type=Path, default=None)
    #parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    #parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    #parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def train():
    print("Training...")

def validate():
    print("Validation...")
  
    
def load_metadata(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data["characters"], data["max_len"]


def load_model_and_weights():
    weights_keras_string = "_weights.keras"
    MODEL_MODEL_PATH = MODEL_NAME
    MODEL_WEIGHT_PATH = MODEL_NAME + weights_keras_string
    model_path = os.path.join(MODEL_DIR_NAME, MODEL_MODEL_PATH)
    model_weight_path = os.path.join(model_path, MODEL_WEIGHT_PATH)
    model_weight_path = "./handwriting/models/model9v3_xl/model9v3_xl_weights.keras"
    model_path = "./handwriting/models/model9v3_xl"
    print(model_path)

    if os.path.exists(model_path):
        print("Loading pre-trained model and weights...")
        model = load_model(model_path)
        model.load_weights(model_weight_path)
        print("Model and weights loaded successfully.")
        return model
    else:
        print("No pre-trained model or weights found.")
        return None

def decode_single_prediction(pred):
    char, max_len = load_metadata("./handwriting/data/meta.pkl")
    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(char), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    result = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    result = tf.gather(result[0], tf.where(tf.math.not_equal(result[0], -1)))
    result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    return result

def infer(model, image):

    img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, 1)
    image = handwriting.preprocess.distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0

    prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
    preds = prediction_model.predict(tf.expand_dims(image, axis=0))
    pred_texts = decode_single_prediction(preds)

    selected_pred_text = pred_texts.replace("|"," ")
    print(f"Prediction: {selected_pred_text}")


def load_json_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config  

def main():
    test_image = './handwriting/data/test.png'
    args = parse_args()
    
    decoder_mapping = {'bestpath': 0,
                       'beamsearch': 1,
                       'wordbeamsearch': 2,
                       'none': 3}
    decoder_type = decoder_mapping[args.decoder]
    print("Choosen decoder: {type}".format(type=args.decoder))
    
    if args.get_config:
        config_file_path = './utils/configs.json'
        config = load_json_config(config_file_path)
        print("Current configuration:")
        print(json.dumps(config, indent=4))
    
    if args.mode == 'infer':
        loaded_model = load_model_and_weights()
        if args.img_file:
            print("Inference with custom Imgage.")
            infer(image=args.img_file, model=loaded_model)
        else:
            infer(image=test_image, model=loaded_model)

    
if __name__ == '__main__':
    main()