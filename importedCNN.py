import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import time

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from skimage.transform import resize

from keras.models import load_model
from keras.utils import to_categorical

import svm
import handTracker


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    loaded_model = load_model('C:/Users/soyon/Documents/Codes/ASL-Translator/ASL.h5')
    loaded_model.summary()

    data = svm.Data()
    X, y = data.load_data(imgdir)

    predictions = loaded_model.predict(X)
    print(classification_report(y, predictions))

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)

        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)

        if tracker.results.multi_hand_landmarks:

            # cropped = tracker.slice_hand_imgs(cv2.flip(image, 1), 0)
            # plt.imshow(cropped)
            # plt.show()

            # lmlist = tracker.find_positions(cv2.flip(image, 1))
            # # print(lmlist)
            # xlist = np.array(lmlist[:, 2])
            # # print(xlist)
            # ylist = np.array(lmlist[:, 3])
            # # print(ylist)
            #
            # xylist = []
            # for cx in xlist:
            #     xylist.append(cx)
            # for cy in ylist:
            #     xylist.append(cy)
            #
            # # xylist = np.stack([xlist, ylist])

            for i in range(len(tracker.results.multi_hand_landmarks)):

                # pos = np.array(xylist[i*(21*2):(i+1)*(21*2)])
                # print(pos)
                # pos = pos.reshape(1, -1)

                # pos_pca = loaded_pca.transform(pos)
                # pos_pca = loaded_pca.inverse_transform(pos_pca)

                cropped = tracker.slice_hand_imgs(cv2.flip(image, 1), i)
                print(cropped.shape)
                cropped = resize(cropped, (64, 64, 3))
                print(cropped.shape)

                predicted_letter = loaded_model.predict(cropped)
                predicted_letter = letters[int(predicted_letter)]
                print(predicted_letter)

                probability = np.ravel(loaded_model.predict_proba(cropped))
                # print(probability)
                probability = max(probability) * 100
                print('confidence:', probability)

                if probability >= 60:
                    tracker.display_letters(frame, i, predicted_letter, round(probability, 2))

        cv2.imshow("Signed English Translator", frame)
        cv2.waitKey(1)
