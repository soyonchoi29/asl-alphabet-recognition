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
from tensorflow.python.keras.utils.np_utils import to_categorical

import cv2
import pickle
import os
import csv

import handTracker


def load_data(datadir):
    index = 0

    folders = sorted(os.listdir(datadir))
    X = []
    y = []
    images = []
    # print(folders)

    tracker = handTracker.HandTracker(max_hands=1)
    # file = open('collected_coordinates_ver2.csv', 'w')
    # writer = csv.writer(file)

    # separate folder for each letter
    for folder in folders:

        print("Loading images from folder", folder, "has started.")
        # imgind = -1

        for image in os.listdir(datadir + '/' + folder):
            # imgind += 1
            # print(imgind)
            #
            # if imgind <= 1300:
            #     continue
            # elif imgind >= 1600:
            #     break

            img = cv2.imread(datadir + '/' + folder + '/' + image)
            # img = resize(img, (64, 64))
            # img = rgb2gray(img)
            # img /= 255
            #
            # X.append(img.flatten())
            # y.append(index)
            img = imread('C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train/A/A1303.jpg')

            # plt.imshow(img)
            # plt.show()

            # print(np.shape(img))
            tracker.find_hands(img)
            # print("Found hand!")

            if tracker.results.multi_hand_landmarks:
                lmlist = tracker.find_positions(img)
                # print(lmlist)

                # for i in range(len(lmlist)):
                #     writer.writerow([image,
                #                      'finger_id = {}'.format(lmlist[i, 1]),
                #                      (lmlist[i, 2], lmlist[i, 3]),
                #                      'Class = {}'.format(index)])

                xlist = np.array(lmlist[:, 2])
                # print(xlist)
                ylist = np.array(lmlist[:, 3])
                # print(ylist)

                xylist = []
                for cx in xlist:
                    xylist.append(cx)
                for cy in ylist:
                    xylist.append(cy)

                # print(xylist)
                # print(len(xylist))

                X.append(xylist)
                y.append(index)
                images.append('{}'.format(image))

        index += 1

    X = np.array(X)
    # print(np.shape(self.X))
    y = np.array(y)
    # print(np.shape(self.y))
    # y = to_categorical(y, len(folders))

    return X, y


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/webcam dataset'
    letters = sorted(os.listdir(imgdir))

    X, y = load_data(imgdir)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    pickle.dump(X_tsne, open('X_tsne_webcam.sav', 'wb'))
    pickle.dump(y, open('y_tsne_webcam.sav', 'wb'))
    print("Saved X_tsne for webcam dataset!")


