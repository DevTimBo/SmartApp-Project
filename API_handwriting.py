# Unsere Klassen
import handwriting.preprocess
import handwriting.load_transfer_data as load_transfer_data
import utils.configs as Config
import API_pipeline as pipeline
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

config_path = r"SmartApp-Project\handwriting\utils\configs.json"
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

#pipeline.main()
loaded_model = pipeline.load_model_and_weights()
image_path = r'SmartApp-Project\handwriting\data\a01-000u-00.png'
pipeline.infer(loaded_model, image_path)

