# import the necessary packages
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

print(tf.__version__)

# load mask detector model
model_name = "mask_detector_test.h5"
maskNet = load_model(model_name)

# original size of the model
convert_bytes(get_file_size(model_name),"MB")

# convert the keras file to tensorflow lite
tf_lite_model_name = "mask_detector.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model(maskNet)
tflite_model = converter.convert()
open(tf_lite_model_name,"wb").write(tflite_model)

# tflite size of the model
convert_bytes(get_file_size(tf_lite_model_name),"MB")
