import tensorflow as tf
from tensorflow import keras
from keras.models import load_model # type: ignore
import numpy as np
import os

#membuat model keras
model = load_model('./breastCNN.h5')

#memuat data uji
def load_testing_data(file_path='./index10_class0.npy'):
    if os.path.exist(file_path):
        try:
            return np.load(file_path)
        except EOFError:
            print("Error")
            return None
    else:
        print("eror not founf")
        return None