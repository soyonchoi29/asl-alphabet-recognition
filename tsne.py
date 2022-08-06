import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import cv2
import pickle
import os
import csv

import handTracker


def load_data(datadir):

    index = 0

    folders = sorted(os.listdir(datadir))
    images = []
    X = []
    y = []
    labels = folders
    # print(folders)

    # tracker = handTracker.HandTracker(max_hands=1)
    # file = open('collected_coordinates.csv', 'w')
    # writer = csv.writer(file)

    # separate folder for each letter
    for folder in folders:

        print("Loading images from folder", folder, "has started.")
        imgind = -1

        for image in os.listdir(datadir + '/' + folder):

            imgind += 1
            # print(imgind)

            if imgind <= 1300:
                continue
            elif imgind >= 1320:
                break

            img = imread(datadir + '/' + folder + '/' + image)
            img = resize(img, (64, 64))
            img = rgb2gray(img)
            img /= 255

            X.append(img.flatten())
            y.append(index)

        index += 1

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    X, y = load_data(imgdir)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tsne = TSNE()
    X_tsne = tsne.fit_transform(X)
    # self.X_pca = self.pca.inverse_transform(self.X_pca)

    pickle.dump(X_tsne, open('X_tsne_kaggle_img.sav', 'wb'))
    pickle.dump(y, open('y_tsne_kaggle_img.sav', 'wb'))
    print("Saved X_tsne for Kaggle dataset!")


