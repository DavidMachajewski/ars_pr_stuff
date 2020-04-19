import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

"""
1. Complete the data.py file and create .csv Dataset
2. Implement a CNN for ex. :
    https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1#1.-Introduction
3. Convert it to .tflite format
4. Check on mobile phone
5. Evtl. help with preprocessing
6. Paar daten annotieren
"""

PATH_TRAIN_DATA = "C:/Users/Natalia/PycharmProjects/ocr/dataset/real_mnist.csv"
PATH_TEST_DATA = "C:/Users/Natalia/PycharmProjects/ocr/dataset/real_mnist_test.csv"

# prepare data
train = pd.read_csv(PATH_TRAIN_DATA)
test = pd.read_csv(PATH_TEST_DATA)

































# THIS IS WITH CSV
# https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1#1.-Introduction

# https://towardsdatascience.com/build-a-multi-digit-detector-with-keras-and-opencv-b97e3cd3b37

# K neirest neighboirhood
# https://qaofficial.com/post/2019/04/16/72849-implementation-of-k-nearest-neighbor-knn-classifier-based-on-tensorflow-taking-mnist-as-an-example.html

# https://github.com/ColinEberhardt/wasm-sudoku-solver/tree/master/training
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# http://ufldl.stanford.edu/housenumbers/
# https://stackoverflow.com/questions/58398983/how-to-improve-digit-recognition-of-a-model-trained-on-mnist
# https://stackoverflow.com/questions/58398983/how-to-improve-digit-recognition-of-a-model-trained-on-mnist
