import os
import cv2
import json
import numpy as np
import pandas as pd
from itertools import chain


class Data:
    def __init__(self):
        print("Provide Labels if you use data_to_csv().")
        self.imgs = []
        self.dataframe = []

    def load_images_from_folder(self, folder):
        """
        folder:
        Files with format: ...
        :return:
        """
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img.ndim > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            if img is not None:
                resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
                self.imgs.append(resized)
        return self.imgs

    def data_to_csv(self, folder, PATH_TO_CSV_FILE, labelsArray=None):
        """
        folder:
        labelsArray:
        Labels have to be
        0: OUT, 1-9: machine, 10: empty, 11 - 19: handwritten
        """
        if labelsArray is None:
            labelsArray = [1 * k for k in range(1, 10) for i in range(0, 915)]
        images = self.load_images_from_folder(folder)
        images = np.array(images)
        FLATTEN_IMAGES = [list(chain.from_iterable(imgMat)) for imgMat in images]  # use this one for PANDAS !!!
        mdigit_labels = labelsArray  # will be changed or advanced
        columns = [str(i) + "x" + str(j) for i in range(1, 29) for j in range(1, 29)]
        index = [str(i) for i in range(0, len(images))]
        # df = pd.DataFrame(data=FLATTEN_IMAGES, index=index, columns=columns)
        df = pd.DataFrame(data=FLATTEN_IMAGES, index=index, columns=columns)
        self.dataframe = df
        df.insert(0, "label", mdigit_labels, True)
        df.to_csv(PATH_TO_CSV_FILE, index=False)
        return df

    def load_json_to_df(self, pathfile):
        with open(pathfile) as f:
            dat = json.load(f)
        df = pd.DataFrame(dat)
        return df













# https://medium.com/bethgelab/ai-still-fails-on-robust-handwritten-digit-recognition-and-how-to-fix-it-a432d84ede18
# https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
# https://github.com/kvfrans/variational-autoencoder
# LINK FIR NN https://medium.com/analytics-vidhya/smart-sudoku-solver-using-opencv-and-tensorflow-in-python3-3c8f42ca80aa
# https://towardsdatascience.com/visualizing-intermediate-activations-of-a-cnn-trained-on-the-mnist-dataset-2c34426416c8
# THIS IS WITH CSV
# https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1#1.-Introduction
