import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import time

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import svm2
import handTracker


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    model = svm2.SVM()
    loaded_model = model.load_model('svm_model_no_pca_world_grid_w_z_coord.sav')

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)
        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)

        if tracker.results.multi_hand_landmarks:

            lmlist = tracker.find_positions(frame)
            # print(lmlist)
            xlist = np.array(lmlist[:, 2])
            # print(xlist)
            ylist = np.array(lmlist[:, 3])
            # print(ylist)
            zlist = np.array(lmlist[:, 4])

            xyzlist = []
            for cx in xlist:
                xyzlist.append(cx)
            for cy in ylist:
                xyzlist.append(cy)
            for cz in zlist:
                xyzlist.append(cz)

            for i in range(len(tracker.results.multi_hand_landmarks)):

                pos = np.array(xyzlist[i*(21*3):(i+1)*(21*3)])
                print(pos)
                pos = pos.reshape(1, -1)

                # pos_pca = loaded_pca.transform(pos)
                # pos_pca = loaded_pca.inverse_transform(pos_pca)

                predicted_letter = loaded_model.predict(pos)
                predicted_letter = letters[int(predicted_letter)]
                print(predicted_letter)

                probability = np.ravel(loaded_model.predict_proba(pos))
                # print(probability)
                probability = max(probability) * 100
                print(probability)

                if probability >= 60:
                    tracker.display_letters(frame, i, predicted_letter, round(probability, 2))

        cv2.imshow("Signed English Translator", frame)
        cv2.waitKey(1)
