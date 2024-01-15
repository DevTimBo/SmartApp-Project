import keras
import tensorflow as tf
from keras import layers
import numpy as np
from keras.models import Sequential
from keras.models import load_model




class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model_default(img_width, img_height, char, model_name):
    # Inputs to the model
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name=model_name)
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model

def build_model_1(img_width, img_height, char, model_name):
    # Inputs to the model
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block with batch normalization
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation=None,
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.BatchNormalization(name="batch_norm1")(x)
    x = keras.layers.Activation("relu", name="relu1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block with batch normalization
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation=None,
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.BatchNormalization(name="batch_norm2")(x)
    x = keras.layers.Activation("relu", name="relu2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs with batch normalization
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.BatchNormalization(name="batch_norm_lstm1")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.BatchNormalization(name="batch_norm_lstm2")(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name=model_name)
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model

def build_model_2(img_width, img_height, char, model_name):
    # Inputs to the model
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block with batch normalization
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation=None,
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.BatchNormalization(name="batch_norm1")(x)
    x = keras.layers.Activation("relu", name="relu1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block with batch normalization
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation=None,
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.BatchNormalization(name="batch_norm2")(x)
    x = keras.layers.Activation("relu", name="relu2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs with batch normalization
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.BatchNormalization(name="batch_norm_lstm1")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.BatchNormalization(name="batch_norm_lstm2")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.BatchNormalization(name="batch_norm_lstm3")(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name=model_name)
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model

def build_model4v2(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model4v2B(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt)
    return model


def build_model4v2v1(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model4v2v2(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.5, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model4v2v2v1(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.5, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.5, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model


def residual_block(x, units):
    # Erster LSTM-Layer
    y = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    y = layers.BatchNormalization()(y)

    # Zweiter LSTM-Layer mit Residual-Verbindung
    y = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(y)
    y = layers.BatchNormalization()(y)

    # Residual-Verbindung
    if x.shape[-1] != y.shape[-1]:
        # Wenn die Dimensionen unterschiedlich sind, füge eine Dense-Schicht hinzu, um sie anzupassen
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    
    y = layers.add([x, y])

    return y

def build_model_with_residual(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Füge Residual-Blöcke hinzu
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    
    return model

def build_model5(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    #x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 4), (img_height // 4) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.5, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

#Mit Batch-Normalization
def build_model4v3(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    # Anpassung der neuen Form basierend auf den Pooling-Operationen
    new_shape = ((img_width // 2), (img_height // 2)*128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_batch_norm")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

#Mit verändertem Pooling-Stride
def build_model4v4(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), strides=(1, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width-1 // 1), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_pooling_stride")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

#Mit GRU anstelle von LSTM
def build_model4v5(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_gru")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

# LSTM mit mehr Neuronen 
def build_model6(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_more_lstm_neurons")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

#Mehr LSTM-Schichten
def build_model7(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_more_lstm_layers")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

# Adamax optimizer
def build_model4v6(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                    
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25, kernel_initializer="glorot_uniform"))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_with_adamax_optimizer")
    opt = keras.optimizers.Adamax()
    model.compile(optimizer=opt)
    return model

def build_model8(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv5")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv6")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.25)(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model8v2(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv5")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv6")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model9(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

def build_model9v2(img_img_width, img_img_height, char):
    input_img = keras.Input(shape=(img_img_width, img_img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.2)(x)
    
    new_shape = ((img_img_width // 4), (img_img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model

# GOAT
def build_model9v3(img_width, img_height, char, lr_value):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x) # from 0.2 to 0.5 
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    

    opt = keras.optimizers.Adam(lr_value)
    model.compile(optimizer=opt)
    
        
    return model

# GRU instead LSTM for 
def build_model9v3_random(img_width, img_height, char, lr_value):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x) # from 0.2 to 0.5 
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    

    opt = keras.optimizers.Adam(learning_rate=lr_value)
    model.compile(optimizer=opt)
    
    return model

def build_model9v3_random_transfer(pretrained_model, img_width, img_height, char, lr_value):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x) # from 0.2 to 0.5 
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    

    opt = keras.optimizers.Adam(learning_rate=lr_value)
    model.compile(optimizer=opt)
    
    return model


def build_model4v2_random(img_width, img_height, char, lr_value):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam(learning_rate=lr_value)
    model.compile(optimizer=opt)
    
    return model

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.5, value_range=(0, 1), seed=42),
        tf.keras.layers.RandomContrast(0.5, seed=42)
    ]
)
def build_model9v3v1A1(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = data_augmentation(input_img)
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt)
    
    return model


img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)
    
def build_model9v3v1A2(img_width, img_height, char, learning_rate_value, seed=None):
    if seed is not None:
        np.random.seed(seed)
        keras.backend.clear_session()


    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = img_augmentation(input_img)

    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    
    # Scale the learning rate
    opt = keras.optimizers.Adam(learning_rate=learning_rate_value)
    model.compile(optimizer=opt)
    
    return model


def build_modelTest(img_width, img_height, char):
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)

    x = keras.layers.Dropout(0.2)(x)
    
    new_shape = ((img_width // 2), (img_height // 2) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    
    return model

def load_and_finetune_model(model, img_width, img_height, char, lr_value):
    # Laden des vorhandenen Modells
    
    for layer in model.layers:
        if "Conv" in layer.name or "dense1" in layer.name:
            layer.trainable = False

    # Hinzufügen von neuen Schichten für die Anpassung an die neue Aufgabe
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(model.get_layer("pool2").output)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(char + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(model.input[1], x)  # Verwende die gleichen Labels wie im Originalmodell

    new_model = keras.models.Model(inputs=model.input, outputs=output, name="finetuned_handwriting_recognizer")

    opt = keras.optimizers.Adam(learning_rate=lr_value)
    new_model.compile(optimizer=opt)

    return new_model


def load_and_finetune_model2(model, img_width, img_height, char, lr_value):
    # Laden des vorhandenen Modells

    # Freeze layers except for the last dense layer
    for layer in model.layers:
        if "dense2" not in layer.name:
            layer.trainable = False

    model.layers.pop()
    x = layers.Dense(char + 2, activation="softmax", name="dense2")(model.get_layer("bidirectional_1").output)

    output = CTCLayer(name="ctc_loss")(model.input[1], x)  # Verwende die gleichen Labels wie im Originalmodell

    new_model = keras.models.Model(inputs=model.input, outputs=output, name="finetuned_handwriting_recognizer")

    opt = keras.optimizers.Adam(learning_rate=lr_value)
    new_model.compile(optimizer=opt)

    return new_model